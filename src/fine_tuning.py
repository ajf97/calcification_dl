import os

import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torchmetrics import JaccardIndex
from torchmetrics.functional import dice

from data.cbis_ddsm import CBISDataset
from data.kios import KIOSDataset
from dvclive import Live
from models.fcn import initialize_model, predict
from utils import segmentation_map


def dfs_freeze(model, freeze=False):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = freeze
        dfs_freeze(child)


def evaluate_model(network, test_dataset, threshold=0.5):
    """Evaluate model on test dataset

    returns:
            metrics: returns a tuple of two arrays. one of the arrays is jaccard index also as known as iou and the other is dice score.
    """

    ious, dice_scores = [], []

    for i, m in test_dataset:
        image, gt_mask = i, m
        prediction = predict(network, image)
        sm = segmentation_map(prediction, threshold)

        pred = torch.tensor(sm)
        gt = segmentation_map(gt_mask, 0.0000001)
        gt = torch.tensor(gt).float()
        jaccard = JaccardIndex(task="binary")

        iou = jaccard(pred / 255, gt / 255)
        dice_score = dice(pred / 255, (gt / 255).int())

        ious.append(iou.item())
        dice_scores.append(dice_score.item())

    metrics = {"iou": np.array(ious).mean(), "dice_score": np.array(dice_scores).mean()}

    return metrics


def train_model(cfg, network, device, inputs, labels, optimizer, criterion):
    # apply some transformations to the image before passing it through the network
    norm = transforms.Normalize(mean=[cfg.sheba_mean], std=[cfg.sheba_std])

    if len(inputs.shape) == 2:
        inputs = inputs[..., None]

    inputs = norm(inputs.float())[:1][None, ...].float()

    labels = labels.expand(1, 1, labels.shape[-2], labels.shape[-1]).float()

    # send the data to gpu
    inputs, labels = inputs.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = network(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    if cfg.train.current_dataset == "kios":
        train_dataset = KIOSDataset(
            cfg.dataset.train_images_path,
            cfg.dataset.train_masks_path,
            transform=True,
            width=cfg.train.width,
        )

        test_dataset = KIOSDataset(
            cfg.dataset.test_images_path,
            cfg.dataset.test_masks_path,
            transform=True,
            width=cfg.train.width,
        )

        if cfg.dataset.validation_images_path != "test":
            validation_dataset = KIOSDataset(
                cfg.dataset.validation_images_path,
                cfg.dataset.validation_masks_path,
                transform=True,
                width=cfg.train.width,
            )
        else:
            validation_dataset = test_dataset

    elif cfg.train.current_dataset == "cbis":
        train_dataset = CBISDataset(
            cfg.dataset.train_images_path,
            cfg.dataset.train_masks_path,
            transform=True,
            width=cfg.train.width,
        )
        test_dataset = CBISDataset(
            cfg.dataset.test_images_path,
            cfg.dataset.test_masks_path,
            transform=True,
            width=cfg.train.width,
        )

        if cfg.dataset.validation_images_path != "test":
            validation_dataset = CBISDataset(
                cfg.dataset.validation_images_path,
                cfg.dataset.validation_masks_path,
                transform=True,
                width=cfg.train.width,
            )
        else:
            validation_dataset = test_dataset
    else:
        raise ValueError("Only two datasets are available: cbis and kios")

    # load the model
    network = initialize_model(cfg.pretrained_weights)

    epochs = cfg.train.epochs

    # define the loss function and optimizer

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        network.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the dataloaders.
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=0,
    )

    # First, we freeze the encoder and unfreezer the last layer
    dfs_freeze(network)
    network.pred.weight.requires_grad = True
    network.pred.weight.bias = True

    print("Weight on the last layer: ", network.pred.weight)
    print("Starting the first training")

    with Live(save_dvc_exp=True) as live:
        live.log_params(cfg.train)
        best_metric = 0.0

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Train the model
                loss = train_model(
                    cfg=cfg,
                    network=network,
                    device=device,
                    inputs=inputs,
                    labels=labels,
                    criterion=criterion,
                    optimizer=optimizer,
                )

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0

                # Evaluation of the model
                eval_dict = {"val": validation_dataset}
                dice_val = 0.0

                for eval, value in eval_dict.items():
                    metrics = evaluate_model(network, value, cfg.train.threshold)

                    for metric_name, value in metrics.items():
                        live.log_metric(metric_name + "_" + eval, value)

                        if eval == "val":
                            dice_val = metrics["dice_score"]

                live.next_step()

                # Save the best model
                if dice_val > best_metric:
                    torch.save(network.state_dict(), cfg.train.model_output)

        print("Weights of pred layer: ", network.pred.weight)

        # Unfreeze all the network and train with a lower lr
        dfs_freeze(network, True)

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            num_workers=0,
        )
        optimizer = optim.SGD(
            network.parameters(), lr=cfg.train.lr / 10, momentum=cfg.train.momentum
        )
        print("Starting the second and last training")

        for epoch in range(epochs // 2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Train the model
                loss = train_model(
                    cfg=cfg,
                    network=network,
                    device=device,
                    inputs=inputs,
                    labels=labels,
                    criterion=criterion,
                    optimizer=optimizer,
                )

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0

                # Evaluation of the model
                eval_dict = {"val": validation_dataset}
                dice_val = 0.0

                for eval, value in eval_dict.items():
                    metrics = evaluate_model(network, value, cfg.train.threshold)

                    for metric_name, value in metrics.items():
                        live.log_metric(metric_name + "_" + eval, value)

                        if eval == "val":
                            dice_val = metrics["dice_score"]

                live.next_step()

                # Save the best model
                if dice_val > best_metric:
                    torch.save(network.state_dict(), cfg.train.model_output)

        metrics = evaluate_model(network, test_dataset, cfg.train.threshold)
        for metric_name, value in metrics.items():
            live.log_metric(metric_name + "_test", value)

        print("Evaluated on test set")
        print(metrics)


# torch.save(network.state_dict(), "../src/models/1.weights")

if __name__ == "__main__":
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    run_experiment()
