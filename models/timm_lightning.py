import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix


class TimmLightning(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        label_smoothing,
        visualise_callback,
        stop_visualise_callback_rounds,
        datamodule_info,
        output_dir="./output",
    ):
        super().__init__()

        self.model = model
        self.opt = optimizer
        self.sch = scheduler

        self.train_acc1 = Accuracy(top_k=1)
        # self.train_acc5 = Accuracy(top_k=5)
        self.eval_acc1 = Accuracy(top_k=1)
        # self.eval_acc5 = Accuracy(top_k=5)

        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.softmax = nn.Softmax(dim=1)

        # Failed Images Callback (Validation Set)
        self.visualise_amt = visualise_callback
        self.stop_rounds = stop_visualise_callback_rounds
        self.counter_rounds = 0

        datamodule_info.setup()  # set up the datamodule to get dataset info too.
        self.num_classes = datamodule_info.num_classes
        self.labels = datamodule_info.eval_dataset.classes
        self.output_dir = output_dir

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):

        if self.sch is not None:
            return [self.opt], [
                {
                    "scheduler": self.sch,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1,
                }
            ]
        else:
            return self.opt

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):

        # Some PL bug for Metric, metric is none and cannot be passed even after it is logged. But it works for torch reducelr.
        # As of now, cannot use LR Scheduler that needs metric in timm schedulers.
        # Reference issue to follow: https://github.com/PyTorchLightning/pytorch-lightning/issues/13260
        scheduler.step(
            epoch=self.current_epoch, metric=metric
        )  # timm's scheduler need the epoch value

    def training_step(self, batch, batch_idx):
        images, target, _ = batch
        output = self.model(images)

        # Target will be list if cutmix is triggered
        if not isinstance(target, list):
            loss_train = self.loss(output, target)
            # update metrics (Training accuracy can only be updated when cutmix is not triggered.)
            self.train_acc1(output, target)
            # self.train_acc5(output, target)
            self.log("train_acc1", self.train_acc1)
            # self.log("train_acc5", self.train_acc5)
        else:
            target1, target2, lam = target
            loss_train = self.loss(output, target1) * lam + self.loss(
                output, target2
            ) * (1 - lam)

        self.log("train_loss", loss_train)

        return {"loss": loss_train}

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target, path = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)

        # update metrics
        self.eval_acc1(output, target)
        # self.eval_acc5(output, target)

        self.log(f"{prefix}_loss", loss_val)
        self.log(
            f"{prefix}_acc1",
            self.eval_acc1,
        )
        # self.log(
        #     f"{prefix}_acc5",
        #     self.eval_acc5,
        # )

        return {
            "loss": loss_val,
            "images": images,
            "target": target,
            "outputs": output,
            "path": path,
        }

    def validation_epoch_end(self, outputs):

        self.counter_rounds += 1

        if (
            self.visualise_amt is False
            or self.visualise_amt == 0
            or self.counter_rounds <= self.stop_rounds
        ):  # Do not want to visualise (Off fail visualiser callback)
            return

        # Stack all the outputs
        all_images = torch.tensor([])
        all_targets = torch.tensor([])
        all_outputs = torch.tensor([])
        all_paths = []

        for out in outputs:
            all_images = torch.cat((all_images, out["images"].cpu()), dim=0)
            all_targets = torch.cat((all_targets, out["target"].cpu()), dim=0)
            all_outputs = torch.cat((all_outputs, out["outputs"].cpu()), dim=0)
            all_paths += out["path"]
        softmax_outputs = torch.amax(self.softmax(all_outputs), dim=1)
        all_targets = all_targets.type(torch.LongTensor)
        all_outputs = torch.argmax(all_outputs, dim=1)
        all_paths = np.array(all_paths)

        failed_prediction_idx = all_targets != all_outputs
        failed_softmax_outputs = softmax_outputs[failed_prediction_idx]
        failed_paths = all_paths[failed_prediction_idx]
        failed_targets = all_targets[failed_prediction_idx]
        failed_outputs = all_outputs[failed_prediction_idx]

        # Visualise failed predictions
        fig = self.create_failed_figure(
            images=all_images,
            actual_targets=all_targets,
            predicted_targets=all_outputs,
            paths=all_paths,
            max_failed_images_visualised=self.visualise_amt
            if self.visualise_amt is not True
            else None,  # give None as flag to visualise all.
        )
        self.logger.experiment.add_figure("Failed Predictions", fig, self.current_epoch)

        # Visualise confusion matrix too (along with visualisation of failed images)
        num_classes = self.num_classes
        labels = self.labels

        confmat = ConfusionMatrix(num_classes=num_classes)
        confusion_matrix = confmat(all_outputs, all_targets)
        per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)

        df_cm = pd.DataFrame(
            confusion_matrix.numpy(),
            index=labels,
            columns=labels,
        )
        size = math.ceil(num_classes**0.5)
        fig_size = max(10, size) + size
        plt.figure(figsize=(fig_size, fig_size))
        sns.set(font_scale=0.5)
        fig_ = sns.heatmap(df_cm, annot=True, cmap="YlGnBu").get_figure()
        self.logger.experiment.add_figure("Confusion Matrix", fig_, self.current_epoch)

        size = math.ceil(num_classes**0.5)
        fig_size = max(10, size) + size
        fig = plt.figure(figsize=(fig_size, fig_size))
        plt.bar([i for i in range(self.num_classes)], per_class_acc)
        plt.title("Accuracy By Class")
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        self.logger.experiment.add_figure("Accuracy By Class", fig, self.current_epoch)

        above_70_idx = failed_softmax_outputs.numpy() > 0.7
        mislabels_path = failed_paths[above_70_idx]
        mislabels_target = failed_targets[above_70_idx]
        mislabels_output = failed_outputs[above_70_idx]

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        df = pd.DataFrame(
            {
                "label": [self.labels[i] for i in mislabels_target],
                "predicted_label": [self.labels[i] for i in mislabels_output],
                "path": mislabels_path,
            }
        )
        df.to_csv(f"{self.output_dir}/mislabels.csv", index=False)

        below_30_idx = failed_softmax_outputs.numpy() < 0.3
        mislabels_path = failed_paths[below_30_idx]
        mislabels_target = failed_targets[below_30_idx]
        mislabels_output = failed_outputs[below_30_idx]
        df = pd.DataFrame(
            {
                "label": [self.labels[i] for i in mislabels_target],
                "predicted_label": [self.labels[i] for i in mislabels_output],
                "path": mislabels_path,
            }
        )
        df.to_csv(f"{self.output_dir}/off-class-labels.csv", index=False)

    def create_failed_figure(
        self,
        images,
        actual_targets,
        predicted_targets,
        paths,
        max_failed_images_visualised,
        shuffle=True,
    ):

        # Find images that model fail to predict correctly
        failed_prediction_idx = actual_targets != predicted_targets
        total_failed = int(sum(failed_prediction_idx))

        failed_images = images[failed_prediction_idx]
        failed_actual_targets = actual_targets[failed_prediction_idx]
        failed_predicted_targets = predicted_targets[failed_prediction_idx]
        failed_paths = paths[failed_prediction_idx]

        # Allow shuffling (validation set is not shuffled)
        if shuffle:
            random_idx = np.random.permutation(total_failed)

            (
                failed_images,
                failed_actual_targets,
                failed_predicted_targets,
                failed_paths,
            ) = (
                failed_images[random_idx],
                failed_actual_targets[random_idx],
                failed_predicted_targets[random_idx],
                failed_paths[random_idx],
            )

        # If user define a maximum number to visualise.
        if max_failed_images_visualised is not None:

            (
                failed_images,
                failed_actual_targets,
                failed_predicted_targets,
                failed_paths,
            ) = (
                failed_images[:max_failed_images_visualised],
                failed_actual_targets[:max_failed_images_visualised],
                failed_predicted_targets[:max_failed_images_visualised],
                failed_paths[:max_failed_images_visualised],
            )

        # Plot figure
        number_to_plot = failed_images.shape[0]  # number of failed images to plot.
        size = math.ceil(number_to_plot**0.5)
        figure_size = max(10, size) + size
        fig = plt.figure(figsize=(figure_size, figure_size))

        # convert to suitable plotting and printing format
        failed_images = failed_images.permute(
            0, 2, 3, 1
        )  # torch format (BCHW)-> matplotlib format (BHWC)
        failed_actual_targets = np.array(failed_actual_targets).astype(int)
        failed_predicted_targets = np.array(failed_predicted_targets).astype(int)

        for idx in range(number_to_plot):
            plt.subplot(size, size, idx + 1)
            plt.imshow(failed_images[idx])
            plt.title(failed_paths[idx].split("/")[-1], fontsize=5)
            plt.ylabel(f"GT: {self.labels[failed_actual_targets[idx]]}", fontsize=5)
            plt.xlabel(
                f"Predict: {self.labels[failed_predicted_targets[idx]]}", fontsize=5
            )
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

        fig.suptitle(
            f"""
            Images that was wrongly predicted by model. Total failed predictions: {total_failed} / {images.shape[0]} .
            Total number of failed images to plot: {number_to_plot}.
            """
        )
        return fig

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")
