import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scl.aug import get_inference_transforms
from scl.custom_datasets import imagenet1k

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Training Loop
def train_log_reg(model, train_loader, criterion, optimizer, num_epochs=10, test_loader=None):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    evaluate(model, test_loader)

def evaluate(model, test_loader, topk=(1, 5, 10)):
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            maxk = max(topk)
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct_expanded = pred.eq(labels.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k = correct_expanded[:k].reshape(-1).float().sum(0, keepdim=True)
                correct[k] += correct_k.item()
                
            total += labels.size(0)
    
    for k in topk:
        accuracy = 100. * correct[k] / total
        print(f'Accuracy of the model on the {total} test images: top-{k} accuracy is {accuracy:.2f}%')



class ImageNetEval:

    def __init__(self, image_size=196):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_sample = 100000
        train_ds = imagenet1k(transform=get_inference_transforms((image_size, image_size)), split="train", sample=n_sample)
        val_ds = imagenet1k(transform=get_inference_transforms((image_size, image_size)), split="validation")
    

        self.train_loader = DataLoader(train_ds,
                                batch_size=64,
                                num_workers=4)
        self.val_loader = DataLoader(val_ds,
                            batch_size=64,
                            num_workers=4)

    @torch.inference_mode
    def evaluate(self, scl_model):
        model = scl_model.target_encoder[0]
        embeddings, labels = self._get_image_embs_labels(model, self.train_loader)
        embeddings_val, labels_val = self._get_image_embs_labels(model, self.val_loader)
        train_features, train_labels = torch.tensor(embeddings).float(), torch.tensor(labels)
        test_features, test_labels = torch.tensor(embeddings_val).float(), torch.tensor(labels_val)
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        model = LinearClassifier(input_dim=2048, num_classes=1000).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-3)
        
        train_log_reg(model, train_loader, criterion, optimizer, num_epochs=10, test_loader=test_loader)

    @torch.inference_mode
    def _get_image_embs_labels(self, model, dataloader):
        embs, labels = [], []
        for _, (images, targets) in enumerate(dataloader):
            with torch.no_grad():
                images = images.to(self.device)
                out = model(images)
                features = out.cpu().detach()
                features = F.normalize(features, p=2, dim=-1)
                embs.extend(features.tolist())
                labels.extend(targets.cpu().detach().tolist())
        return np.array(embs), np.array(labels)
