import torch
from torchmetrics.functional.classification.accuracy import accuracy
from trainer import MyCustomTrainer

import os
import lightning as L


class MNISTModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            # fully connected layer, output 10 classes
            torch.nn.Linear(32 * 7 * 7, 10),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        accuracy_train = accuracy(logits.argmax(-1), y, num_classes=10, task="multiclass", top_k=1)

        return {"loss": loss, "accuracy": accuracy_train}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        return [optim], [{
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", verbose=True),
            "monitor": "val_accuracy",
            "interval": "epoch",
            "frequency": 1,
        }]

    def validation_step(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)


def train(model):
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    train_set = MNIST(root="/tmp/data/MNIST", train=True, transform=ToTensor(), download=True)
    val_set = MNIST(root="/tmp/data/MNIST", train=False, transform=ToTensor(), download=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=64, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4
    )

    # MPS backend currently does not support all operations used in this example.
    # If you want to use MPS, set accelerator='auto' and also set PYTORCH_ENABLE_MPS_FALLBACK=1
    accelerator = "cpu" if torch.backends.mps.is_available() else "auto"

    nodes = os.environ.get("NODES_SIZE", 1)
    gpus_per_node = os.environ.get("GPUS_PER_NODE", 2)
    # trainer = L.Trainer(accelerator="gpu", devices=gpus_per_node, num_nodes=nodes, strategy="ddp" , enable_model_summary=True,
    #                     accumulate_grad_batches=2,limit_train_batches=10, limit_val_batches=20, max_epochs=10,
    #                     log_every_n_steps=5)
    trainer = MyCustomTrainer(
        accelerator=accelerator, devices=gpus_per_node, nodes=nodes, grad_accum_steps=1,limit_train_batches=10,
        limit_val_batches=20, max_epochs=3
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
        # to fix kubeflow torch operator problem, where it can set RANK as Node_rank according to replica
    # and WORLD_SIZE as the sum of replica, which supports only one gpu per pod and wrong NODE_RANK.
    if os.environ.get("EX_WORLD_SIZE", None):
       os.environ["WORLD_SIZE"] = os.environ.pop("EX_WORLD_SIZE")
       os.environ["NODE_RANK"] = os.environ.pop("RANK")
    torch.set_float32_matmul_precision('high')
    torch.set_float32_matmul_precision('high')
    train(MNISTModule())
