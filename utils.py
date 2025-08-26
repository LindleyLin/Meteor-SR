import numpy as np
import torch
import yaml

from torch.optim.lr_scheduler import _LRScheduler

# 区分训练测试集
def traintestsplit(dataX, datay, ratio, is_shuffle=True):
    n_samples = len(dataX)
    indices = list(range(n_samples))
    if is_shuffle:
        np.random.shuffle(indices)
    trainX = dataX[indices][:int(ratio * n_samples)]
    trainy = datay[indices][:int(ratio * n_samples)]
    testX = dataX[indices][int(ratio * n_samples):]
    testy = datay[indices][int(ratio * n_samples):]
    return trainX, trainy, testX, testy

# 加载数据集
def get_data(config):
    dataX = np.fromfile('dataset/trainx', dtype='float32')
    dataX = dataX.reshape((-1, config["channels"], config["piy"], config["pix"]))
    dataX = torch.tensor(dataX, dtype=torch.float32)

    dataY = np.fromfile(f'dataset/trainy_{config["upscale"]}', dtype='float32')
    dataY = dataY.reshape((-1, config["channels"], config["piy"] * config["upscale"], config["pix"] * config["upscale"]))
    dataY = torch.tensor(dataY, dtype=torch.float32)

    return dataX, dataY

def normalize0(dataX, dataY):
    dataX = torch.log1p(dataX)
    dataY = torch.log1p(dataY)

    dataX = (dataX - dataX.min()) / (dataX.max() - dataX.min())
    dataY = (dataY - dataY.min()) / (dataY.max() - dataY.min())
    return dataX, dataY

def normalize_1(dataX, dataY):
    dataX = torch.log1p(dataX)
    dataY = torch.log1p(dataY)

    meanX = dataX.mean()
    stdX = dataX.std()
    meanY = dataY.mean()
    stdY = dataY.std()

    dataX = (dataX - meanX) / stdX
    dataY = (dataY - meanY) / stdY

    return dataX, dataY

def get_loader(norm_option=1):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    dataX, dataY = get_data(config)

    if norm_option == 1:
        dataX, dataY = normalize_1(dataX, dataY)
    else:
        dataX, dataY = normalize0(dataX, dataY)

    trainX, trainy, testX, testy = traintestsplit(dataX, dataY, config["ratio"], config["is_shuffle"])

    traindata = torch.utils.data.TensorDataset(trainX, trainy)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=config["batch_size"], shuffle=config["is_shuffle"])
    testdata = torch.utils.data.TensorDataset(testX, testy)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=config["batch_size"], shuffle=config["is_shuffle"])

    return train_loader, test_loader

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)