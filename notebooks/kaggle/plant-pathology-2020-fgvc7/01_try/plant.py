import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from efficientnet_pytorch import EfficientNet

from typing import Callable, List, Tuple, Dict
from pathlib import Path

import catalyst
from catalyst.utils import imread
from catalyst.dl import utils
from catalyst.utils import split_dataframe_train_test
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose
from catalyst.data.augmentor import Augmentor
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, CheckpointCallback, \
    EarlyStoppingCallback

from transformers import AdamW

from collections import defaultdict, OrderedDict
from tqdm import tqdm
from torchsummary import summary
DictDataset = Dict[str, object]


def create_dataset(root_dir: str, mask: str, config) -> DictDataset:
    dataset = defaultdict(list)
    images = list(Path(config.root_images).glob(mask))
    for image in sorted(images):
        label = image.stem
        dataset[label] = str(image)

    return dataset


def create_dataframe(dataset: DictDataset, **dataframe_args) -> pd.DataFrame:
    data = [(key, value) for key, value in dataset.items()]
    df = pd.DataFrame(data, **dataframe_args)
    return df


def get_loaders(train_transforms_fn, valid_transforms_fn, config, batch_size: int = 8, num_workers: int = 20,
                sampler=None) -> OrderedDict:
    train_data, valid_data = get_datasets(config)

    open_fn = ReaderCompose([
        ImageReader(
            input_key="filepath",
            output_key="features",
            rootpath=config.root_images
        ),
        ScalarReader(
            input_key="disease_type",
            output_key="targets",
            default_value=-1,
            dtype=np.int64
        ),
        ScalarReader(
            input_key="disease_type",
            output_key="targets_one_hot",
            default_value=-1,
            dtype=np.int64,
            one_hot_classes=config.num_classes
        )
    ])

    train_loader = utils.get_loader(
        train_data,
        open_fn=open_fn,
        dict_transform=train_transforms_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=sampler is None,  # shuffle data only if Sampler is not specified (PyTorch requirement)
        sampler=sampler,
        drop_last=True,
    )

    valid_loader = utils.get_loader(
        valid_data,
        open_fn=open_fn,
        dict_transform=valid_transforms_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=None,
        drop_last=True,
    )

    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


def get_model(model_name: str, num_classes: int, pretrained: str = "imagenet") -> EfficientNet:
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    return model


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = A.Compose([item for sublist in transforms_to_compose for item in sublist])
    return result


def get_datasets(config):
    train_df = pd.read_csv(config.root + 'train.csv')
    test_df = pd.read_csv(config.root + 'test.csv')
    dataset = create_dataset(root_dir=config.root_images, mask="Train*.jpg", config=config)
    df_path = create_dataframe(dataset, columns=["image_id", "filepath"])
    df_with_labels = pd.merge(df_path, train_df, left_on='image_id', right_on='image_id')
    df_with_labels["disease_type"] = df_with_labels["healthy"] * 0 + df_with_labels["multiple_diseases"] * 1 + \
                                     df_with_labels["rust"] * 2 + df_with_labels["scab"] * 3
    df_with_labels.head(10)

    train_data, valid_data = split_dataframe_train_test(df_with_labels, test_size=0.3, random_state=config.seed)
    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict('records')

    return train_data, valid_data