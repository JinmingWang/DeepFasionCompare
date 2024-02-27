from Dataset.DFCDataset import DFCDataset, collateFunc
from torch.utils.data import DataLoader
from Models.CompareModel import CompareModel
from torch import nn
from torch import optim
import torch
from typing import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime


def train():

    bs = 64
    init_lr = 1e-3
    n_epochs = 100


    # Initialize dataset
    dataset = DFCDataset("E:/Data/deepfasion/train_test_256/pth_train")
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collateFunc)

    # Initialize model
    model = CompareModel().cuda()
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    # Initialize tensorboard
    log_dir = "Runs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    # Train
    for epoch in range(n_epochs):
        for img1, img2, label in tqdm(dataloader, desc=f"Epoch {epoch}"):
            # img1: (B, 3, 256, 192)
            # img2: (B, 3, 256, 192)
            # label: (B, 1)

            # pred: (B, 1)
            pred= model(img1, img2)

            # loss: (1)
            loss = loss_func(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss", loss, global_step=epoch)
        torch.save(model.state_dict(), f"{log_dir}/model.pth")
