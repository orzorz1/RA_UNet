import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from data.load import *
from torch.utils.data import DataLoader
import torch
from models.RA_UNet_1 import RA_UNet_1
from modules.functions import dice_loss
import math
from commons.plot import print2D,printXandY
print(torch.cuda.is_available())
def train_step_1():
    batch_size = 10
    epochs = 10
    model = RA_UNet_1()
    try:
        model.load_state_dict(torch.load("RA_UNet_1.pth", map_location='cpu'))
    except FileNotFoundError:
        print("模型不存在")
    else:
        print("加载模型成功")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(20):
        print("训练进度：{index}/20".format(index=i+1))
        dataset = dataset_step_1(i)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        for epoch in range(epochs):
            # training-----------------------------------
            model.train()
            train_loss = 0
            train_acc = 0
            for batch, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
                out = model(batch_x)
                printXandY(out.detach().cpu().numpy()[0,0,:,:]*255, batch_y.detach().cpu().numpy()[0,0,:,:]*255)
                loss = dice_loss(out, batch_y)
                train_loss += loss.item()
                print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f'
                      % (epoch + 1, epochs, batch + 1, math.ceil(len(dataset) / batch_size),loss.item(),))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Train Loss: %.6f' % (train_loss / (math.ceil(len(dataset) / batch_size))))
        torch.save(model.state_dict(), "RA_UNet_1.pth")


if __name__ == '__main__':
    train_step_1()