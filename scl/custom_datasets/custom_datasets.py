from datasets import load_dataset
import random

from scl.custom_datasets.huggingface_dataset import HuggingfaceDataset


def tiny_imagenet(transform, split="train"):
    data = load_dataset('Maysee/tiny-imagenet', split=split)
    return HuggingfaceDataset(data, transform)


def food101(transform, split="train", sample=-1):
    data = load_dataset('food101', split=split)
    random.seed(42)
    if sample > 0:
        indices = random.sample(range(0, len(data)), sample)
        data = data.select(indices)
    return HuggingfaceDataset(data, transform)


def imagenet1k(transform, split="train", sample=-1):
    data = load_dataset("imagenet-1k", split=split)
    random.seed(42)
    if sample > 0:
        indices = random.sample(range(0, len(data)), sample)
        data = data.select(indices)
    return HuggingfaceDataset(data, transform)
