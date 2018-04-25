import argparse

import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils._models import *
from utils.read_data import *


class Predict(object):
    def __init__(self, args):
        self.args = args
        self.loader = self.__init_loader()
        self.model = self.__load_model()

    def predict(self):
        self.model.eval()

        progress_bar = tqdm(self.loader, desc='Predicting')
        pred = np.array([])
        image_names = np.array([])

        for (data, names) in progress_bar:
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output = self.model(data)

            if len(pred):
                pred = np.concatenate([pred, np.array(output.data)])
                image_names = np.concatenate([image_names, np.array(names)])
            else:
                pred = np.array(output.data)
                image_names = np.array(names)
        self.__save_array(pred, 'pred')
        self.__save_array(image_names, 'image_names')

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
            model = CheXNet()
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
            model = CheXNet()

        if self.args.cuda:
            model = torch.nn.DataParallel(model).cuda()

        return model

    @staticmethod
    def __init_transform():
        transform_list = [transforms.Resize(224),
                          transforms.RandomCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])]
        return transforms.Compose(transform_list)

    def __init_loader(self):
        test_loader = DataLoader(
            ChestXrayDataSetForPredicting(data_dir=self.args.data_dir,
                                          transforms=self.__init_transform()),
            batch_size=self.args.batch_size,
            shuffle=False
        )
        return test_loader

    def __save_array(self, array, name):
        np.savez('./result/{}_{}.npz'.format(self.args.weight_dir, name), array)


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='PyTorch To test Chest-Xray by using densenet')
    parser.add_argument('--model', type=str, default='DenseNet',
                        help='The model name [DenseNet121, DenseNet161, DenseNet169, '
                             'DenseNet201, CheXNet, ResNet18, ResNet34, ResNet50,'
                             ' ResNet101, ResNet152, VGG191]')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='the path of the data directory')
    parser.add_argument('--weight-dir', type=str, default='',
                        help='the path of trained model')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='the batch size when testing')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--reshape-size', type=int, default=224,
                        help='the size of the input image')
    parser.add_argument('--classes', type=int, default=156,
                        help='the #classes of target')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    predictor = Predict(args=args)
    predictor.predict()
