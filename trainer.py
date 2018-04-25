import time

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import callbacks
from utils._models import *
from utils.read_data import ChestXrayDataSet


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = self.__load_model()
        self.train_loader, self.val_loader = self.__init_loader()
        self.callbacks = self.__init_callback()
        self.optimizer = self.__init_optimizer()
        self.scheduler = self.__init_scheduler()
        self.loss = self.__init_loss()
        self.loss_val = self.__init_min_loss()

    def train(self):
        self.callbacks.on_train_begin()
        for epoch in range(1, self.args.epochs + 1):
            self.callbacks.on_epoch_begin(epoch)
            self.__epoch_train()
            self.scheduler.step(self.__epoch_val(epoch))
        self.callbacks.on_train_end()

    def __epoch_train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.callbacks.on_batch_begin(batch_idx)
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            batch_logs = {'loss': np.array(loss.data[0]),
                          'size': np.array(len(target)),
                          'batch': np.array(batch_idx)}
            self.callbacks.on_batch_end(batch_idx, batch_logs)

    def __epoch_val(self, epoch):
        self.model.eval()
        loss_val = 0
        progress_bar = tqdm(self.val_loader, desc='Validation')

        for (data, target) in progress_bar:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            loss_val += self.loss(output, target)
        loss_min = self.__save_model(loss_val.data[0], epoch)
        epoch_logs = {'val_loss': np.array(loss_val.data)}
        self.callbacks.on_epoch_end(epoch=epoch, logs=epoch_logs)
        progress_bar.write('Epoch: {} - validation results - '
                           'Total val_loss: {:.4f} '
                           '- Min val_loss: {} '
                           '- Learning rate: {}'.format(epoch,
                                                        loss_val.data[0],
                                                        loss_min,
                                                        self.optimizer.param_groups[
                                                            0]['lr']))
        return loss_val.data[0]

    def __init_loader(self):
        train_loader = DataLoader(
            ChestXrayDataSet(data_dir=self.args.data_dir,
                             file_list=self.args.train_csv,
                             transforms=self.__init_transform()),
            batch_size=self.args.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            ChestXrayDataSet(data_dir=self.args.data_dir,
                             file_list=self.args.val_csv,
                             transforms=self.__init_transform()),
            batch_size=self.args.val_batch_size,
            shuffle=True
        )
        return train_loader, val_loader

    def __load_model(self):
        if self.args.model == 'VGG19':
            model = VGG19(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'DenseNet121':
            model = DenseNet121(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'DenseNet161':
            model = DenseNet161(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'DenseNet169':
            model = DenseNet169(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'DenseNet201':
            model = DenseNet201(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'CheXNet':
            model = CheXNet(classes=self.args.classes)
        elif self.args.model == 'ResNet18':
            model = ResNet18(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'ResNet34':
            model = ResNet34(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'ResNet50':
            model = ResNet50(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'ResNet101':
            model = ResNet101(pretrained=self.args.pretrained, classes=self.args.classes)
        elif self.args.model == 'ResNet152':
            model = ResNet152(pretrained=self.args.pretrained, classes=self.args.classes)
        else:
            model = DenseNet121(pretrained=self.args.pretrained, classes=self.args.classes)

        if self.args.cuda:
            model = torch.nn.DataParallel(model).cuda()

        if self.args.weight_dir:
            model.load_state_dict(torch.load(self.args.weight_dir)['state_dict'])
        print("Load {} Model".format(self.args.model))
        return model

    def __init_callback(self):
        callback_params = {'epochs': self.args.epochs,
                           'samples': len(self.train_loader) * self.args.batch_size,
                           'steps': len(self.train_loader),
                           'metrics': {'acc': np.array([]),
                                       'loss': np.array([]),
                                       'val_acc': np.array([]),
                                       'val_loss': np.array([])}}
        callback_list = callbacks.CallbackList(
            [callbacks.BaseLogger(),
             callbacks.TQDMCallback(),
             ])
        callback_list.set_params(callback_params)
        callback_list.set_model(self.model)
        return callback_list

    def __init_optimizer(self):
        return optim.Adam(params=self.model.parameters(), lr=self.args.lr, eps=1e-08, weight_decay=1e-5)

    def __init_transform(self):
        transform_list = [transforms.Resize(self.args.reshape_size),
                          transforms.RandomCrop(self.args.crop_size),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])]
        return transforms.Compose(transform_list)

    def __init_loss(self):
        return torch.nn.BCELoss(size_average=True)

    def __init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        return scheduler

    def __save_model(self, val_loss, epoch_id):
        if val_loss < self.lossMin:
            self.lossMin = val_loss
            file_name = './models/m-{}-{}.pth.tar'.format(self.__get_date(),
                                                          self.args.model)
            torch.save({'epoch': epoch_id + 1,
                        'state_dict': self.model.state_dict(),
                        'best_loss': self.lossMin,
                        'optimizer': self.optimizer.state_dict()},
                       file_name)
        return self.lossMin

    def __get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def __init_min_loss(self):
        loss_val = 100000
        if self.args.weight_dir:
            loss_val = torch.load(self.args.weight_dir)['best_loss']
        return loss_val
