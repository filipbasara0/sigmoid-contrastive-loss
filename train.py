import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import multiprocessing
import random
import numpy as np
import math
from tqdm.auto import tqdm
from torchinfo import summary

from scl import ReLIC, scl_loss

from scl.utils import accuracy, get_dataset, get_encoder
from scl.stl10_eval import STL10Eval

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def cosine_scaling_gamma(step, start_gamma=2.0, end_gamma=0.0, total_steps=20000):
    """
    Scales gamma value from start_gamma to end_gamma over total_steps using a cosine schedule.
    
    Parameters:
    - step (int): Current training step.
    - total_steps (int): Total number of steps over which to scale gamma.
    - start_gamma (float): Starting value of gamma.
    - end_gamma (float): Ending value of gamma.
    
    Returns:
    - float: Scaled gamma value for the current step.
    """
    if step > total_steps:
        return end_gamma
    
    cosine = np.cos(np.pi * step / total_steps) / 2.0 + 0.5
    return cosine * (start_gamma - end_gamma) + end_gamma


# cosine EMA schedule (increase from beta_base to one) as defined in https://arxiv.org/abs/2010.07922
# k -> current training step, K -> maximum number of training steps
def update_beta(k, K, beta_base):
    k = torch.tensor(k, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)

    beta = 1 - (1 - beta_base) * (torch.cos(torch.pi * k / K) + 1) / 2
    return beta.item()


def train_scl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modify_model = True if "cifar" in args.dataset_name else False
    encoder = get_encoder(args.encoder_model_name, modify_model)

    init_tau, init_b, max_tau = np.log(5), -5, 5
    scl_model = ReLIC(encoder,
                        mlp_out_dim=args.proj_out_dim,
                        mlp_hidden=args.proj_hidden_dim,
                        init_tau=init_tau, init_b=init_b)

    if args.ckpt_path:
        model_state = torch.load(args.ckpt_path)
        scl_model.load_state_dict(model_state)
    scl_model = scl_model.to(device)

    summary(scl_model, input_size=[(1, 3, 128, 128), (1, 3, 128, 128)])

    params = list(scl_model.online_encoder.parameters()) + [scl_model.tau, scl_model.b]
    optimizer = torch.optim.Adam(params,
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    ds = get_dataset(args)
    train_loader = DataLoader(ds,
                              batch_size=args.batch_size,
                              num_workers=multiprocessing.cpu_count() - 8,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True,
                              shuffle=True)

    scaler = GradScaler(enabled=args.fp16_precision)

    stl10_eval = STL10Eval()
    total_num_steps = (len(train_loader) *
                       (args.num_epochs + 2)) - args.update_beta_after_step
    beta = args.beta
    global_step = 0
    total_loss = 0.0
    init_conf_penalty, conf_pen_total_steps = args.gamma, args.gamma_scaling_steps
    n_global, n_local = args.num_global_views, args.num_local_views
    if args.mini_batch_size > 0 and args.batch_size // args.mini_batch_size > 0:
        mini_batch_size = args.mini_batch_size
    else:
        mini_batch_size = args.batch_size
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, (views, labels) in enumerate(progress_bar):
            global_views = views[:n_global]
            local_views = views[n_global:n_global + n_local]

            if args.use_gamma_scaling:
                conf_penalty = cosine_scaling_gamma(global_step, init_conf_penalty, 0.0, conf_pen_total_steps)
            else:
                conf_penalty = init_conf_penalty

            num_mini_batches = args.batch_size // mini_batch_size
            loss = 0
            invariance_loss = 0
            for mini_batch_idx in range(num_mini_batches):
                start_idx = mini_batch_idx * mini_batch_size
                end_idx = start_idx + mini_batch_size
                mini_batch_global_views = [v[start_idx:end_idx].to(device) for v in global_views]
                mini_batch_local_views = [v[start_idx:end_idx].to(device) for v in local_views]

                with autocast(enabled=args.fp16_precision):
                    projections_online = []
                    projections_target = []
                    for view in mini_batch_global_views:
                        projections_online.append(scl_model.get_online_pred(view))
                        projections_target.append(scl_model.get_target_pred(view))
                    for view in mini_batch_local_views:
                        projections_online.append(scl_model.get_online_pred(view))
                    mini_batch_loss = 0
                    # invariance_loss var used only for logging
                    mini_batch_invariance_loss = 0
                    scale = 0
                    for i_t, target_pred in enumerate(projections_target):
                        for i_o, online_pred in enumerate(projections_online):
                            if i_t != i_o:
                                scl_loss_, invar_loss = scl_loss(online_pred, target_pred,
                                                                scl_model.tau, scl_model.b, 
                                                                args.alpha, max_tau=max_tau,
                                                                gamma=conf_penalty)
                                mini_batch_loss += scl_loss_
                                mini_batch_invariance_loss += invar_loss
                                scale += 1

                    mini_batch_loss /= scale
                    mini_batch_loss = mini_batch_loss / num_mini_batches
                    mini_batch_invariance_loss /= scale
                    mini_batch_invariance_loss = mini_batch_invariance_loss / num_mini_batches
                    loss += mini_batch_loss
                    invariance_loss += mini_batch_invariance_loss

                    scaler.scale(mini_batch_loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if global_step > args.update_beta_after_step and global_step % args.update_beta_every_n_steps == 0:
                scl_model.update_params(beta)
                beta = update_beta(global_step, total_num_steps, args.beta)

            if global_step <= args.update_beta_after_step:
                scl_model.copy_params()

            total_loss += loss.item()
            epoch_loss += loss.item()
            avg_loss = total_loss / (global_step + 1)
            ep_loss = epoch_loss / (step + 1)

            epoch_kl_loss += invariance_loss.item()
            ep_kl_loss = epoch_kl_loss / (step + 1)

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.num_epochs} | "
                f"Step {global_step+1} | "
                f"Epoch Loss: {ep_loss:.4f} |"
                f"Total Loss: {avg_loss:.4f} |"
                f"KL Loss: {ep_kl_loss:.6f} |"
                f"Conf Penalty: {conf_penalty:.3f} |"
                f"EMA Beta: {beta:.6f} |"
                f"Alpha: {args.alpha:.3f} |"
                f"Temp: {scl_model.tau.exp().item():.3f} |"
                f"Bias: {scl_model.b.item():.3f} |"
                f"Lr: {current_lr:.6f}")

            global_step += 1
            if global_step % args.log_every_n_steps == 0:
                with torch.no_grad():
                    x, x_prime = projections_online[-2], projections_target[-1]
                    x, x_prime = F.normalize(x, p=2, dim=-1), F.normalize(x_prime, p=2, dim=-1)
                    logits = torch.mm(x, x_prime.t()) * scl_model.tau.exp().clamp(0, max_tau) + scl_model.b
                labels = torch.arange(logits.size(0)).to(logits.device)
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                print("#" * 100)
                print('acc/top1 logits1', top1[0].item())
                print('acc/top5 logits1', top5[0].item())
                print("#" * 100)

                torch.save(scl_model.state_dict(),
                           f"{args.save_model_dir}/scl_model.pth")
                scl_model.save_encoder(f"{args.save_model_dir}/encoder.pth")

            if global_step % (args.log_every_n_steps * 5) == 0:
                stl10_eval.evaluate(scl_model)
                print("!" * 100)
