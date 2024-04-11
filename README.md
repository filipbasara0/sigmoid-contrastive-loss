
# Sigmoid Contrastive Learning on Images

A PyTorch implementation of the sigmoid pairwise loss for contrastive self-supervised learning on images.


## Results

Models are pretrained on training subsets - for `CIFAR10` 50,000 and for `STL10` 100,000 images. For evaluation, I trained and tested LogisticRegression on frozen features from:
1. `CIFAR10` - 50,000 train images
2. `STL10` - features were learned on 100k unlabeled images. LogReg was trained on 5k train images and evaluated on 8k test images.

Linear probing was used for evaluating on features extracted from encoders using the scikit LogisticRegression model.

More detailed evaluation steps and results for [STL10](https://github.com/filipbasara0/relic/blob/main/notebooks/linear-probing-stl.ipynb) can be found in the notebooks directory. 

| Dataset       | Architecture| Encoder   | Feature dim | Proj. head dim | Batch size | Epochs | Gamma         | Top1 % |
|---------------|-------------|-----------|-------------|----------------|------------|--------|---------------|--------|
| STL10         | ReLIC       | ResNet-50 | 2048        | 64             | 256        | 100    | 1.0           | 85.42  |
| STL10         | ReLIC       | ResNet-50 | 2048        | 64             | 256        | 100    | 1.0 + schedule| 86.06  |
| ImageNet-1k   | ReLIC       | ResNet-50 | 2048        | 64             | 128        | 50     | 1.0           | 59.42  |

[Here](https://drive.google.com/file/d/1XaZBdvPGPh2nQzzHAJ_oL41c1f8Lc_FN/view?usp=sharing) is a link to a resnet50 encoder trained on the ImageNet-1k subset on a single GPU, for only 50 epochs and a batch size of 128. It achieved 59.42% Top 1 accuracy, 83.8% Top-5 accuracy and 89.6% Top-10 accuracy after training linear regresssion on frozen feautures from the training dataset.

## About the project

The training architecture consists of an online and a target encoder (EMA) with a simple critic MLP projector and is based on [Representation Learning via Invariant Causal Mechanisms (ReLIC)](https://arxiv.org/abs/2010.07922). The loss function is a sigmoid constrastive loss adapted from [SigLIP](https://arxiv.org/abs/2303.15343), with an addition of a confidence penalty gamma that balances the ratio of positive and negative samples per batch, amplifying learning from harder examples and improving training stability. Loss function also supports a KL divergence regularization term that acts as an invariance penalty and forces the representations to stay invariant under data augmentations and amplifies intra-class distances.

When using larger batch sizes (eg. larger than 128), it is possible and recommended to enable gamma scheduling, which acts as a form of curriculum learning, balancing the learning from positive and negative samples during early stages, enabling faster convergence and better overall results. Initially, we are learning more from positive samples, but as training progresses the signal from negative samples becomes more prevalent. The result of training for 100 epochs on STL-10 can be seen in the table below, where gamma is specified as `1.0 + schedule`. During that run, gamma is initialized at 1.0 and decayed to 0.0 over 20_000 steps using the cosine schedule.


Repo includes the multi-crop augmentation and extends the loss function is extended to support an arbitrary number of small (local) and large (global) views. Using this technique generally results in more robust and higher quality representations.

## Usage

### Instalation

```bash
$ pip install sigmoid-contrastive-learning
```

Code currently supports ResNet18, ResNet50 and an experimental version of the EfficientNet model. Supported datasets are STL10, CIFAR10 and ImageNet-1k.

All training is done from scratch.

### Examples
`CIFAR10` ResNet-18 model was trained with this command:

`python run_training.py --dataset_name "cifar10" --encoder_model_name resnet18 --fp16_precision --beta 0.99 --alpha 1.0`

`STL10` ResNet-50 model was trained with this command:

`python run_training.py --dataset_name "stl10" --encoder_model_name resnet50 --fp16_precision  --beta 0.99 --gamma 1.0 --gamma_scaling_steps 20_000 --use_gamma_scaling`

### Detailed options
Once the code is setup, run the following command with optinos listed below:
`scl_train [args...]⬇️`

```
Sigmoid Contrastive Learning

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path where datasets will be saved
  --dataset_name {stl10,cifar10,tiny_imagenet,food101,imagenet1k}
                        Dataset name
  -m {resnet18,resnet50,efficientnet}, --encoder_model_name {resnet18,resnet50,efficientnet}
                        model architecture: resnet18, resnet50 or efficientnet (default: resnet18)
  -save_model_dir SAVE_MODEL_DIR
                        Path where models
  --num_epochs NUM_EPOCHS
                        Number of epochs for training
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -mb MINI_BATCH_SIZE, --mini_batch_size MINI_BATCH_SIZE
                        Optionally split batch views into mini batches. Similarily to gradient accumulation, enables (significantly) larger batch sizes.Could be further utilized for parallel or distributed mini batch processing, which would require multiple GPUs (may add at
                        some point).
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
  --fp16_precision      Whether to use 16-bit precision for GPU training
  --proj_out_dim PROJ_OUT_DIM
                        Projector MLP out dimension
  --proj_hidden_dim PROJ_HIDDEN_DIM
                        Projector MLP hidden dimension
  --log_every_n_steps LOG_EVERY_N_STEPS
                        Log every n steps
  --beta BETA           Initial EMA coefficient
  --alpha ALPHA         Regularization loss factor
  --update_beta_after_step UPDATE_BETA_AFTER_STEP
                        Update EMA beta after this step
  --update_beta_every_n_steps UPDATE_BETA_EVERY_N_STEPS
                        Update EMA beta after this many steps
  --gamma GAMMA         Initial confidence penalty
  --gamma_scaling_steps GAMMA_SCALING_STEPS
                        Number of first N steps during which gamma will be scaled from the inital value to 0
  --use_gamma_scaling   Whether to use gamma(conf penalty) cosine scaling
  --ckpt_path CKPT_PATH
                        Specify path to scl_model.pth to resume training
  --num_global_views NUM_GLOBAL_VIEWS
                        Number of global (large) views to generate through augmentation
  --num_local_views NUM_LOCAL_VIEWS
                        Number of local (small) views to generate through augmentation
```

## Citation

```
@misc{basara2024sigmoid,
  author = {Basara, Filip},
  title = {Sigmoid Contrastive Learing for Vision},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/filipbasara0/sigmoid-contrastive-learning}}
}

@misc{mitrovic2020representation,
      title={Representation Learning via Invariant Causal Mechanisms}, 
      author={Jovana Mitrovic and Brian McWilliams and Jacob Walker and Lars Buesing and Charles Blundell},
      year={2020},
      eprint={2010.07922},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{zhai2023sigmoid,
      title={Sigmoid Loss for Language Image Pre-Training}, 
      author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
      year={2023},
      eprint={2303.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
