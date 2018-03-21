from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from read_data import ChestXrayDataSet
from densenet import DenseNet121
from torch.autograd import Variable
from tqdm import tqdm
import torch
import numpy as np


class ChexnetTester(object):
    def __init__(self, args):
        self.args = args
        self.loader = self.__init_loader()
        self.model = self.__load_model()
        self.loss = self.__init_loss()

    def test(self):
        self.model.eval()

        progress_bar = tqdm(self.loader, desc='Testing')
        test_loss = 0
        y_true = np.array([])
        y_pred = np.array([])

        for (data, target) in progress_bar:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += self.loss(output, target)

            if len(y_true):
                y_true = np.concatenate([y_true, np.array(target.data)])
                y_pred = np.concatenate([y_pred, np.array(output.data)])
            else:
                y_true = np.array(target.data)
                y_pred = np.array(output.data)
        self.__save_array(y_true, 'y_true')
        self.__save_array(y_pred, 'y_pred')

    def __load_model(self):
        model = DenseNet121(self.args.classes)
        # model = torch.nn.DataParallel(model).cuda()

        model.load_state_dict(torch.load(self.args.model_dir)['state_dict'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_transform(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transform_list = [transforms.Resize(self.args.reshape_size),
                          transforms.ToTensor(),
                          normalize]
        return transforms.Compose(transform_list)

    def __init_loader(self):
        test_loader = DataLoader(
            ChestXrayDataSet(data_dir=self.args.data_dir,
                             file_list=self.args.test_csv,
                             transforms=self.__init_transform()),
            batch_size=self.args.batch_size,
            shuffle=False
        )
        return test_loader

    def __init_loss(self):
        return torch.nn.BCELoss(size_average=True)

    def __save_array(self, array, name):
        np.savez(array, '{}.npz'.format(name))
