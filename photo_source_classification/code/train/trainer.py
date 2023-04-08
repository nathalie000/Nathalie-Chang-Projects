# This file define how to train a model
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np

TRAINER_DEBUG = False

class Trainer:
    def __init__(self, model, optimizer, num_epoch, loss, train_dataset, val_dataset, batch_size, device) -> None:
        """Initilization of the trainer
        
        Args:
            model: (what type???), the model to be trained
            optimizer: (???), the optimizer to be used in the training process
            num_epoches: int, the number of epoches to run the training process
            lr: float / double, learning rate
            training_dataset: (???), training dataset
            val_dataset: (???), validation dataset
        """
        self.model = model
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.loss = loss
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        
        if train_dataset != None:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        else:
            self.train_loader = None
        
        if val_dataset != None:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            self.train_loader = None

        self.device = device
    
    def prediction(self, data, label) -> None:
        ans = []

        if TRAINER_DEBUG:
            print(data)
            print(label)

        # for i in range(len(data)):
        #     if((data[i]>=0.5 and label[i] == 1) or (data[i]<0.5 and label[i] == 0)):
        #         ans.append(1)
        #     else:
        #         ans.append(0)
        # ans = np.array(ans)
        ans = np.equal((np.round(data + 1) - 1), label).astype(int)

        return ans

    def train(self) -> None:
        self.model.to(self.device)
        for epoch in range(self.num_epoch):
            self.one_train_process(epoch)
    
    def one_train_process(self, epoch) -> None:
        """ Define how to train model for one time"""
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        self.model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = self.model(data[0].to(self.device)) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = self.loss(train_pred, data[1].to(self.device)) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
            self.optimizer.step() # 以 optimizer 用 gradient 更新參數值

            train_acc += np.sum(self.prediction(train_pred.cpu().data.numpy(), data[1].numpy()))
            train_loss += batch_loss.item()

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                val_pred = self.model(data[0].to(self.device))
                batch_loss = self.loss(val_pred, data[1].to(self.device))
                val_acc += np.sum(self.prediction(val_pred.cpu().data.numpy(), data[1].numpy()))
                val_loss += batch_loss.item()

            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, self.num_epoch, time.time()-epoch_start_time, \
             train_acc/self.train_dataset.__len__(), train_loss/self.train_dataset.__len__(), val_acc/self.val_dataset.__len__(), val_loss/self.val_dataset.__len__()))

    
    def validation(self) -> None:
        """ Define how to make the validation based on the validation dataset"""
        val_acc = 0.0
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                val_pred = self.model(data[0].to(self.device))
                batch_loss = self.loss(val_pred, data[1].to(self.device))
                val_acc += np.sum(self.prediction(val_pred.cpu().data.numpy(), data[1].numpy()))
                val_loss += batch_loss.item()
                
            print('Val Acc: %3.6f loss: %3.6f' % (val_acc/self.val_dataset.__len__(), val_loss/self.val_dataset.__len__()))

    # Define anything that are needed...