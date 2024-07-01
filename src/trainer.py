import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import os

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


class ConfusionMatrix():
    def __init__(self, labels):
        self.labels = np.array(labels)
        self.cls_num = len(self.labels)
        # row == True : col == Pred
        self.matrix = np.array([[0]*self.cls_num for _ in range(self.cls_num)])

        self.true = []
        self.pred = []


    def __call__(self, output, target):
        return self.calc_confmat(output, target)


    def calc_confmat(self, outputs, targets):
        # заполнение матрицы ошибок новыми данными
        output_classes = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        target_classes = targets.cpu().detach().numpy()
        self.matrix[target_classes, output_classes] += 1
        
        # для дальнейшего 'classification_report()'
        self.true = [*self.true, *target_classes]
        self.pred = [*self.pred, *output_classes]


    def classification_report(self, output_dict=False):
        true = self.labels[self.true]
        pred = self.labels[self.pred]
        cls_report = classification_report(
            true,
            pred,
            labels=self.labels,
            output_dict=output_dict,
        )
        return cls_report


    def plot(self):
        figsize = (8, 8)
        fig, ax = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay(self.matrix, display_labels=self.labels)
        disp.plot(ax=ax)
        plt.show()


class EarlyStopper:
    def __init__(self, model: nn.Module, weights_name, scripted_name, patience=10, delta=3):
        self.model = model
        self.weights_name = weights_name
        self.scripted_name = scripted_name

        self.patience = patience
        self.delta = delta

        self.counter = 0

        self.best_loss = float('inf')
        self.best_f1 = 0.0


    def __call__(self, valid_loss, valid_f1=0.0):
        match self.early_stop(valid_loss, valid_f1):
            case 0:
                print(f'\nEarly stopped: best_loss = {self.best_loss:.3} | best_f1 = {self.best_f1:.3}')
                print(f'Model weights saved with name  "{self.weights_name}"')
                print(f'Scripted model saved with name "{self.scripted_name}"')
                return True
            case 2:
                self.save_model()
                return False
            case _:
                return False


    def save_model(self):
        torch.save(self.model, self.weights_name)
        try:
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(self.scripted_name)
        except RuntimeError: # конвертированные модели не скриптуются
            pass


    def early_stop(self, valid_loss, valid_f1):
        """
        0: stop
        1: continue
        2: save best
        """
        valid_loss = float(f'{valid_loss:.{self.delta}}')
        valid_f1 = float(f'{valid_f1:.{self.delta}}')
        if valid_f1 > self.best_f1:
            self.best_f1 = valid_f1
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
            self.counter = 0
            return 2
        elif valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.counter = 0
            return 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return 0

        return 1
    

    from tqdm import tqdm

class Trainer():
    '''
    Classification task trainer. Should be reimplemented through a base class
    '''
    def __init__(self, model: nn.Module, train_loader, valid_loader, test_loader, name: str, labels=None):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)

        self.weights_name = f'{name}_weights.pt'
        self.scripted_name = f'{name}_scripted.pt'

        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader

        self.class_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()

        self.train_losses = []
        self.train_f1s = []

        self.valid_losses = []
        self.valid_f1s = []

        self.early_stopper = EarlyStopper(
                self.model,
                self.weights_name,
                self.scripted_name,
                patience=5,
                delta=3,
        )

        self.labels = labels
        if self.labels == None:
            model.eval()
            self.labels = list(range(len(train_loader.dataset[0][1])))

        self.confmat_test: ConfusionMatrix

    def get_best_model(self):
        if os.path.exists(self.scripted_name):
            print(f'loaded best from disk {self.scripted_name}')
            return torch.jit.load(self.scripted_name)
        return self.model


    def train(self, epoch_num, lr=0.001, load_from_disk=False):
        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                # betas=(0.999, 0.9999),
                # weight_decay=0.01,
        )
        self.early_stopper.counter = 0

        if load_from_disk:
            self.model = self.get_best_model()
        else:
            if os.path.exists(self.weights_name): os.remove(self.weights_name)
            if os.path.exists(self.scripted_name): os.remove(self.scripted_name)

        self.model.to(self.device)
        for epoch in range(epoch_num):
            print(f'\nEpoch: {epoch+1} / {epoch_num}')

            train_loss, train_f1 = self.train_step(self.train_loader)
            self.train_losses.append(train_loss)
            self.train_f1s.append(train_f1)

            valid_loss, valid_f1 = self.valid_step(self.valid_loader)
            self.valid_losses.append(valid_loss)
            self.valid_f1s.append(valid_f1)

            if self.early_stopper(valid_loss, valid_f1):
                break


    def test(self):
        print('\nTest-Epoch')
        self.model = self.get_best_model()
        self.model.to(self.device)
        confmat = self.valid_step(self.test_loader, return_confmat=True)

        report = confmat.classification_report(output_dict=False)
        if confmat.cls_num > 20:
            report_list = report.split('\n')
            report = '\n'.join([*report_list[:2], *report_list[-3:]])
        print('\n', report)
        return confmat


    def train_step(self, train_loader):
        self.model.train()
        run_loss = 0.0
        confmat = ConfusionMatrix(labels=self.labels)

        for img, target in tqdm(train_loader):
            self.optimizer.zero_grad()
            img, target = img.to(self.device), target.to(self.device)

            output = self.model(img)#.reshape(target.shape)
            loss = self.class_criterion(output, target)
            loss.backward()
            self.optimizer.step()

            run_loss += loss.item()
            confmat(output, target)
            

        train_loss = run_loss / len(train_loader)
        train_f1 = confmat.classification_report(output_dict=True)['weighted avg']['f1-score']
        print(f'  train loss:\t {train_loss:.3}\t | train f1:\t {train_f1:.3}')

        return train_loss, train_f1


    def valid_step(self, valid_loader, return_confmat=False) -> (tuple | ConfusionMatrix):
        self.model.eval()
        run_loss = 0.0
        confmat = ConfusionMatrix(labels=self.labels)

        for img, target in tqdm(valid_loader):
            img, target = img.to(self.device), target.to(self.device)

            output = self.model(img)#.reshape(target.shape)

            run_loss += self.class_criterion(output, target).item()
            confmat(output, target)

        valid_loss = run_loss / len(valid_loader)
        valid_f1 = confmat.classification_report(output_dict=True)['weighted avg']['f1-score']
        print(f'  valid loss:\t {valid_loss:.3}\t | valid f1:\t {valid_f1:.3}')

        if return_confmat:
            return confmat
        return valid_loss, valid_f1

    def plot(self):
        epochs = list(range(1, len(self.train_losses)+1))

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        axes[0].plot(epochs, self.train_losses, '--b',label='train')
        axes[0].plot(epochs, self.valid_losses, 'r',label='valid')
        axes[0].set(xlabel='epoch num', ylabel='loss', title='Loss')
        axes[0].grid()
        axes[0].legend()

        axes[1].plot(epochs, self.train_f1s, '--b', label='train')
        axes[1].plot(epochs, self.valid_f1s, 'r', label='valid')
        axes[1].set(xlabel='epoch num', ylabel='f1', title='F1-score')
        axes[1].grid()
        axes[1].legend()

        fig.tight_layout()
        plt.show() 