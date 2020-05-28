import warnings

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
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

class ConfigExperiment:
    logdir = "./logs/experiment_02"
    submission_file = "experiment_02.csv"
    seed = 42
    batch_size = 8
    model_name = 'efficientnet-b0'
    size = 512
    root_images = "../../../data/raw/plant-pathology-2020-fgvc7/images/"
    root = "../../../data/raw/plant-pathology-2020-fgvc7/"
    num_classes = 4
    patience = 5
    num_epochs = 50
    lr = 0.003
    class_names = ["healthy", "multiple_diseases", "rust", "scab"]
    is_fp16_used = False


def create_dataset(root_dir: str, mask: str) -> DictDataset:
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


def get_loaders(open_fn: Callable, train_transforms_fn, valid_transforms_fn, batch_size: int = 8, num_workers: int = 20,
                sampler=None) -> OrderedDict:
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



def pre_transforms(image_size=224):
    result = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0)
    ]
    return result


def hard_transforms():
    result = [
        # Random shifts, stretches and turns with a 50% probability
        A.RandomResizedCrop(height=config.size, width=config.size, p=1.0),
        A.Flip(),
        A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

        # Pixels
        A.OneOf([
            A.IAAEmboss(p=1.0),
            A.IAASharpen(p=1.0),
            A.Blur(p=1.0),
        ], p=0.5),

        # Affine
        A.OneOf([
            A.ElasticTransform(p=1.0),
            A.IAAPiecewiseAffine(p=1.0)
        ], p=0.5),
    ]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [A.Normalize(p=1.0), ToTensorV2(p=1.0), ]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = A.Compose([item for sublist in transforms_to_compose for item in sublist])
    return result


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

config = ConfigExperiment()
config.size = EfficientNet.get_image_size(config.model_name)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
utils.set_global_seed(config.seed)
utils.prepare_cudnn(deterministic=True)

train_df = pd.read_csv(config.root + 'train.csv')
test_df = pd.read_csv(config.root + 'test.csv')
dataset = create_dataset(root_dir=config.root_images, mask="Train*.jpg")
df_path = create_dataframe(dataset, columns=["image_id", "filepath"])
df_with_labels = pd.merge(df_path, train_df, left_on='image_id', right_on='image_id')
df_with_labels["disease_type"] = df_with_labels["healthy"] * 0 + df_with_labels["multiple_diseases"] * 1 + \
                                 df_with_labels["rust"] * 2 + df_with_labels["scab"] * 3
df_with_labels.head(10)

train_data, valid_data = split_dataframe_train_test(df_with_labels, test_size=0.3, random_state=config.seed)
train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict('records')

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

train_transforms = compose([
    pre_transforms(config.size),
    hard_transforms(),
    post_transforms()
])
valid_transforms = compose([
    pre_transforms(config.size),
    post_transforms()
])

show_transforms = compose([
    pre_transforms(config.size),
    hard_transforms()
])

train_data_transforms = Augmentor(
    dict_key="features",
    augment_fn=lambda x: train_transforms(image=x)["image"]
)
valid_data_transforms = Augmentor(
    dict_key="features",
    augment_fn=lambda x: valid_transforms(image=x)["image"]
)

loaders = get_loaders(
    open_fn=open_fn,
    train_transforms_fn=train_data_transforms,
    valid_transforms_fn=valid_data_transforms,
    batch_size=config.batch_size,
)

model = get_model(config.model_name, config.num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience, verbose=True, mode="min",
                                                       factor=0.3)

device = utils.get_device()

if config.is_fp16_used:
    fp16_params = dict(opt_level="O1")  # params for FP16
else:
    fp16_params = None

runner = SupervisedRunner(device=device)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    # We can specify the callbacks list for the experiment;
    # For this task, we will check accuracy, AUC and F1 metrics
    callbacks=[
        AccuracyCallback(num_classes=config.num_classes),
        AUCCallback(
            num_classes=config.num_classes,
            input_key="targets_one_hot",
            class_names=config.class_names
        ),
        F1ScoreCallback(
            input_key="targets_one_hot",
            activation="Softmax"
        ),
        CheckpointCallback(
            save_n_best=1,
            #             resume_dir="./models/classification",
            metrics_filename="metrics.json"
        ),
        EarlyStoppingCallback(
            patience=config.patience,
            metric="auc/_mean",
            minimize=False
        )
    ],
    # path to save logs
    logdir=config.logdir,

    num_epochs=config.num_epochs,

    # save our best checkpoint by AUC metric
    main_metric="auc/_mean",
    # AUC needs to be maximized.
    minimize_metric=False,

    # for FP16. It uses the variable from the very first cell
    fp16=fp16_params,

    # prints train logs
    verbose=True
)
