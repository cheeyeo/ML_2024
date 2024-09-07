import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import pytorch_warmup as warmup
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
    # MODEL CONFIG FOR PIT model based on paper
    img_size = 32
    patch_size = 1
    num_hiddens = 192
    mlp_num_hiddens = 768
    num_heads = 8
    num_blks = 12
    emb_dropout = 0.1
    blk_dropout = 0.1
    lr = 0.004
    weight_decay = 0.3
    batch_size = 64
    # batch_size = 1024
    num_classes = 100
    epochs = 2400

    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    trainset = CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transforms
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    testset = CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transforms
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # visualize some training data
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(images.shape)
    imgs = images[:4]
    print(imgs.shape)
    print(labels)

    show(torchvision.utils.make_grid(imgs), stats[0], stats[1])

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

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    print(optimizer)

    warmup_steps = len(trainloader) * 20
    num_steps = len(trainloader) * epochs - warmup_steps
    print(f"NUM STEPS: {num_steps}, WARMUP STEPS: {warmup_steps}")

    # Linear warmup as in paper
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_steps)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    for epoch in range(epochs):
        train_loss = 0.0

        model.train()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            optimizer.zero_grad()

            yhat = model(inputs)
            loss = criterion(yhat, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with warmup_scheduler.dampening():
                # print("INSIDE WARMUP DAMPENING...")
                # print(warmup_scheduler.last_step)
                if warmup_scheduler.last_step + 1 >= warmup_steps:
                    print('WARMUP ENDED. Running LR scheduler')
                    lr_scheduler.step()

            # print(f"Running step: {i}")
             # print statistics
            if i % 256 == 0:    # print every 2000 mini-batches
                print(f'[Epoch: {epoch + 1}, {i + 1:5d}] loss: {train_loss}')
        
        print(f"Epoch: {epoch + 1}, Loss: {train_loss}, Train Loss: {train_loss / len(trainloader):.2f}")