import time
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import TrainData
from network import TeacherNet, StudentNet


class Trainer:
    def __init__(self, config, log_dir, distil_weight=0.5):
        '''Initialize the varibles for training
        Args:
            log_dir: (pathlib.Path) the direction used for logging
        '''
        self.max_temp = config['max_temp']
        self.min_temp = config['min_temp']
        self.distil_weight = distil_weight
        self.log_dir = log_dir
        self.training_result_dir = self.log_dir / 'training_result'
        if not self.training_result_dir.exists():
            self.training_result_dir.mkdir(parents=True)
        self.set_logger()

        # Datasets and dataloaders
        self.train_set = TrainData('dataset', 'birds.csv', 'train', aug=config['aug'])
        self.valid_set = TrainData('dataset', 'birds.csv', 'valid', aug=False)
        
        self.train_loader = DataLoader(self.train_set, config['batch_size'], shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_set, config['batch_size'], shuffle=False, num_workers=4)

        self.device = 'cuda'

        # teacher
        self.teacher_model = TeacherNet().to(self.device)
        self.teacher_optimizer = torch.optim.Adam(self.teacher_model.parameters(), lr=config['teacher_lr'])
        self.teacher_scheduler = torch.optim.lr_scheduler.StepLR(self.teacher_optimizer, step_size=2, gamma=0.6)
        self.teacher_max_epoch = config['teacher_epoch']
        
        # student
        self.student_model = StudentNet().to(self.device)
        self.student_optimizer = torch.optim.Adam(self.student_model.parameters(), lr=config['student_lr'])
        if config['student_scheduler'] == 'step':
            self.student_scheduler = torch.optim.lr_scheduler.StepLR(self.student_optimizer, step_size=1, gamma=0.9)
        else:
            self.student_scheduler = torch.optim.lr_scheduler.CyclicLR(self.student_optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=1, step_size_down=14, mode='triangular2', cycle_momentum=False)
        self.student_max_epoch = config['student_epoch']
        
        self.criterion = nn.CrossEntropyLoss()
        self.steps_per_epoch = len(self.train_loader)

        # config
        for key, value in config.items():
            log = f'{key:30}: {value}'
            self.logger.info(log)
        self.logger.info('')

    def set_logger(self):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        file_handler = logging.FileHandler(str(self.training_result_dir / 'log.txt'), mode='w')
        console_handler = logging.StreamHandler()
        self.logger = logging.getLogger()
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_temp(self):
        t = self.epoch / (self.max_epoch - 1) # 0 ~ 1
        return (1-t) * self.max_temp + t * self.min_temp

    def run(self, option):
        self.start_time = time.time()
        metrics = {'train_loss': [], 'valid_loss': [], 'train_f1_score': [], 'valid_f1_score': [], 'learning_rate': []}
        self.max_epoch = self.teacher_max_epoch if option == 'teacher' else self.student_max_epoch

        for self.epoch in range(self.max_epoch): # epochs
            self.temp = self.get_temp()
                
            if option == 'teacher':
                lr = get_lr(self.teacher_optimizer)
                log = f'lr: {lr}\n'
                train_loss, train_f1_score = self.train_teacher() # train 1 epoch               
            else:
                lr = get_lr(self.student_optimizer)
                log = f'lr: {lr}\n'
                train_loss, train_f1_score = self.train_student() # train 1 epoch
            valid_loss, valid_f1_score = self.valid(option) # valid 1 epoch
            
            log += f'Epoch: {self.epoch}/{self.max_epoch}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}, Train f1 score: {train_f1_score:.3f}, Valid f1 score: {valid_f1_score:.3f}, Time: {(time.time() - self.start_time):.2f}' 
            self.logger.info(log)
            
            metrics['train_loss'].append(train_loss)
            metrics['valid_loss'].append(valid_loss)
            metrics['train_f1_score'].append(train_f1_score)
            metrics['valid_f1_score'].append(valid_f1_score)
            metrics['learning_rate'].append(lr)
            
            # Save the parameters(weights) of the model to disk
            if torch.tensor(metrics['valid_f1_score']).argmax() == self.epoch:
                if option == 'teacher':
                    torch.save(self.teacher_model.state_dict(), str(self.training_result_dir / f'teacher_model.pth'))
                else:
                    torch.save(self.student_model.state_dict(), str(self.training_result_dir / f'student_model.pth'))

        # Plot the loss curve against epoch
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.set_title('Loss')
        ax.plot(range(self.epoch + 1), metrics['train_loss'], label='Train')
        ax.plot(range(self.epoch + 1), metrics['valid_loss'], label='Valid')
        ax.legend()
        plt.show()
        fig.savefig(str(self.training_result_dir / f'{option}_loss.jpg'))
        plt.close()

        # Plot the f1 score curve against epoch
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.set_title('F1 score')
        ax.plot(range(self.epoch + 1), metrics['train_f1_score'], label='Train')
        ax.plot(range(self.epoch + 1), metrics['valid_f1_score'], label='Valid')
        ax.legend()
        plt.show()
        fig.savefig(str(self.training_result_dir / f'{option}_f1_score.jpg'))
        plt.close()

        # Plot the learning rate curve against epoch
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.set_title('Learning rate')
        ax.plot(range(self.epoch + 1), metrics['learning_rate'])
        plt.show()
        fig.savefig(str(self.training_result_dir / f'{option}_learning_rate.jpg'))
        plt.close()

    def train_teacher(self):
        '''Train one epoch
        1. Switch model to training mode
        2. Iterate mini-batches and do:
            a. clear gradient
            b. forward to get loss
            c. loss backward
            d. update parameters
        3. Return the average loss in this epoch
        '''
        self.teacher_model.train()
        loss_steps = []
        gt_cls = []
        pred_cls = []

        for step, (img_b, cls_b) in enumerate(iter(self.train_loader)):
            img_b = img_b.to(self.device)
            cls_b = cls_b.to(self.device)
            
            self.teacher_optimizer.zero_grad()
            pred_b = self.teacher_model(img_b)

            # step 1 caculate the loss
            loss = self.criterion(pred_b, cls_b)
            # step 2 back propagation
            loss.backward()
            # step 3 update parameters using optimizer
            self.teacher_optimizer.step()

            if step % 200 == 0:
                log = f'Epoch: {self.epoch}/{self.max_epoch}, Step: {step}/{self.steps_per_epoch}, Train loss: {loss:.3f}, Time: {(time.time() - self.start_time):.2f}'
                self.logger.info(log)
                
            # loss
            loss_steps.append(loss.detach().item())

            # record class for calculating f1 score
            _, pred_cls_b = pred_b.max(1)
            pred_cls.append(pred_cls_b.cpu())
            gt_cls.append(cls_b.cpu())
            
        self.teacher_scheduler.step()
        
        avg_loss = sum(loss_steps) / len(loss_steps)
        f1 = f1_score(torch.cat(gt_cls), torch.cat(pred_cls), average='weighted')
        return avg_loss, f1

    def train_student(self):
        '''Train one epoch
        1. Switch model to training mode
        2. Iterate mini-batches and do:
            a. clear gradient
            b. forward to get loss
            c. loss backward
            d. update parameters
        3. Return the average loss in this epoch
        '''
        self.student_model.train()
        loss_steps = []
        gt_cls = []
        pred_cls = []

        for step, (img_b, cls_b) in enumerate(iter(self.train_loader)):
            img_b = img_b.to(self.device)
            cls_b = cls_b.to(self.device)
            
            self.student_optimizer.zero_grad()
            student_pred_b = self.student_model(img_b)
            with torch.no_grad():
                teacher_pred_b = self.teacher_model(img_b)

            # step 1 caculate the loss
            loss = self.calculate_kd_loss(student_pred_b, teacher_pred_b, cls_b)
            # step 2 back propagation
            loss.backward()
            # step 3 update parameters using optimizer
            self.student_optimizer.step()

            if step % 200 == 0:
                log = f'Epoch: {self.epoch}/{self.max_epoch}, Step: {step}/{self.steps_per_epoch}, Train loss: {loss:.3f}, Time: {(time.time() - self.start_time):.2f}'
                self.logger.info(log)
                
            # loss
            loss_steps.append(loss.detach().item())

            # record class for calculating f1 score
            _, pred_cls_b = student_pred_b.max(1)
            pred_cls.append(pred_cls_b.cpu())
            gt_cls.append(cls_b.cpu())
            
        self.student_scheduler.step()
        
        avg_loss = sum(loss_steps) / len(loss_steps)
        f1 = f1_score(torch.cat(gt_cls), torch.cat(pred_cls), average='weighted')
        return avg_loss, f1

    @torch.no_grad()
    def valid(self, option):
        '''Validate one epoch
        1. Switch model to evaluation mode and turn off gradient (by @torch.no_grad() or with torch.no_grad())
        2. Iterate mini-batches and do forwarding to get loss
        3. Return average loss in this epoch
        '''
        if option == 'teacher':
            self.teacher_model.eval()
        else:
            self.student_model.eval()
        loss_steps = []
        gt_cls = []
        pred_cls = []

        for img_b, cls_b in iter(self.valid_loader):
            img_b = img_b.to(self.device)
            cls_b = cls_b.to(self.device)
            
            teacher_pred_b = self.teacher_model(img_b)
            if option == 'teacher':
                loss = self.criterion(teacher_pred_b, cls_b)
                _, pred_cls_b = teacher_pred_b.max(1)
            else:
                student_pred_b = self.student_model(img_b)
                loss = self.calculate_kd_loss(student_pred_b, teacher_pred_b, cls_b)
                _, pred_cls_b = student_pred_b.max(1)
            
            # loss
            loss_steps.append(loss.detach().item())
            
            # record class for calculating f1 score
            pred_cls.append(pred_cls_b.cpu())
            gt_cls.append(cls_b.cpu())
            
        avg_loss = sum(loss_steps) / len(loss_steps)
        f1 = f1_score(torch.cat(gt_cls), torch.cat(pred_cls), average='weighted')
        return avg_loss, f1

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        
        # hard
        hard_loss = F.cross_entropy(y_pred_student, y_true)
   
        # soft
        soft_student_out = F.log_softmax(y_pred_student / self.temp, dim=1)
        soft_teacher_out = F.softmax(y_pred_teacher / self.temp, dim=1)
        soft_loss = F.kl_div(soft_student_out, soft_teacher_out, reduction='batchmean')
        
        loss = (1 - self.distil_weight) * hard_loss + (self.temp ** 2) * self.distil_weight * soft_loss

        return loss
    

    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
