import warnings
import os
import catalyst
from catalyst.utils import imread
from catalyst.dl import utils
from catalyst.utils import split_dataframe_train_test
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose
from catalyst.data.augmentor import Augmentor
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback, CheckpointCallback, \
    EarlyStoppingCallback
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import plant
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


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


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.filterwarnings('ignore')
    os.environ["PYTHONWARNINGS"] = "ignore"
    config = ConfigExperiment()
    config.size = EfficientNet.get_image_size(config.model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    utils.set_global_seed(config.seed)
    utils.prepare_cudnn(deterministic=True)


    train_transforms = plant.compose([
        pre_transforms(config.size),
        hard_transforms(),
        post_transforms()
    ])
    valid_transforms = plant.compose([
        pre_transforms(config.size),
        post_transforms()
    ])

    show_transforms = plant.compose([
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

    loaders = plant.get_loaders(
        train_transforms_fn=train_data_transforms,
        valid_transforms_fn=valid_data_transforms,
        batch_size=config.batch_size,
        config=config
    )

    model = plant.get_model(config.model_name, config.num_classes)

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
