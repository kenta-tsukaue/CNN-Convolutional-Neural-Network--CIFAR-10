"""
======参考サイト=====
(一番参考にしたサイト)
https://xminatolog.com/post/135


(たたみこみ　→　バッチ正規化　→　畳み込み　→　バッチ正規化　→　レル　→　ドロップアウト　がいい組み合わせ)
https://qiita.com/02130515/items/278bc7fe0b3d7dd2e7ae


"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=4 * 4 * 128, out_features=num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5,))
    ])

    # 前処理（学習用）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(10),                                
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    validation_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    model = CNN(10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    num_epocs = 15
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epocs):
        # 学習
        running_loss = 0.0
        running_acc = 0.0
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            running_acc += torch.mean(pred.eq(labels).float()).item()
            optimizer.step()
        running_loss /= len(train_dataloader)
        running_acc /= len(train_dataloader)
        losses.append(running_loss)
        accs.append(running_acc)
        
        # 検証
        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_imgs, val_labels in validation_dataloader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)
            val_output = model(val_imgs)
            val_loss = criterion(val_output, val_labels)
            val_running_loss += val_loss.item()
            val_pred = torch.argmax(val_output, dim=1)
            val_running_acc += torch.mean(val_pred.eq(val_labels).float()).item()
        val_running_loss /= len(validation_dataloader)
        val_running_acc /= len(validation_dataloader)
        val_losses.append(val_running_loss)
        val_accs.append(val_running_acc)
        print("epoch: {}, loss: {}, acc: {}\n" \
        "val_epoch: {}, val_loss: {}, val_acc: {}".format(epoch, running_loss, running_acc, epoch, val_running_loss, val_running_acc))


    fig = plt.figure()
    ax = fig.add_subplot()
    plt.style.use('ggplot')
    ax.plot(losses, label='train loss')
    ax.plot(val_losses, label='val loss')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("/public/tsukaue/study_party/CNN/loss of cifar10 classification.png")

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.style.use('ggplot')
    ax.plot(accs, label='train acc')
    ax.plot(val_accs, label='val acc')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.savefig("/public/tsukaue/study_party/CNN/acc of cifar10 classification.png")
