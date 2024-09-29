import numpy as np
import lightning as L
from lightning import Fabric
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torchinfo
from vit.model import ViT


def show(imgs, means, stds):
    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.detach()
        
        means = torch.tensor(means).reshape(3, 1, 1)
        stds = torch.tensor(stds).reshape(3, 1, 1)
        img = img * stds + means

        # print(means.shape)
        # print(f"IMG: {img.shape}")

        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    fig.savefig("test.png")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    L.seed_everything(123)

    # MODEL CONFIG FOR PIT model based on paper
    img_size = 32
    patch_size = 1
    num_hiddens = 192
    mlp_num_hiddens = 768
    num_heads = 12
    num_blks = 12
    emb_dropout = 0.1
    blk_dropout = 0.1
    lr = 0.004
    weight_decay = 0.3
    batch_size = 128
    # batch_size = 1024 # actual batch size used in paper...
    num_classes = 100
    warmup_epochs = 20
    epochs = 2400

    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    trainset = CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=train_transforms
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testset = CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=test_transforms
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Setup Fabric
    fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    
    fabric.launch()

    trainloader, testloader = fabric.setup_dataloaders(
        trainloader, testloader
    )

    # visualize some training data
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imgs = images[:4]
    print(imgs.shape)
    print(labels.shape)

    # show(torchvision.utils.make_grid(imgs), stats[0], stats[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViT(
        img_size,
        patch_size,
        num_hiddens,
        mlp_num_hiddens,
        num_heads,
        num_blks,
        emb_dropout,
        blk_dropout,
        num_classes=num_classes
    )

    torchinfo.summary(model, input_size=(batch_size, 3, 32, 32))

    model = torch.compile(model)
    model.to(device)

    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

    model, optimizer = fabric.setup(model, optimizer)

    criterion = nn.CrossEntropyLoss()

    # Linear warmup as in paper
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        warmup_start_lr=lr,
        eta_min=1e-6
    )

    for epoch in range(epochs):
        train_loss = 0.0

        model.train()
        for i, data in enumerate(trainloader, 0):
            model.train()
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            optimizer.zero_grad()

            yhat = model(inputs)
            loss = criterion(yhat, labels)
            fabric.backward(loss)
            optimizer.step()

            train_loss += loss.item()

            if i > 0 and i % 256 == 0:    # print every 256 mini-batches
                print(f'[Epoch: {epoch + 1}, {i}] Batch loss: {train_loss / i:.2f}')
        
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss / len(trainloader):.2f}")

        # Run eval
        with torch.no_grad():
            model.eval()

            test_acc = MulticlassAccuracy(num_classes=num_classes).to(device)

            for (features, targets) in testloader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                prediction = torch.argmax(outputs, 1)
                test_acc.update(prediction, targets)
            
            fabric.print(test_acc.compute())
            fabric.print(f"Epoch: {epoch+1}, Acc: {test_acc.compute() * 100}%")
        
        scheduler.step()