import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils._models import *
from utils.read_data import ChestXrayDataSet


class Tester(object):
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

        # model.load_state_dict(torch.load('./models/m-{}.pth.tar'.format(self.args.weight_dir))['state_dict'])
        return model

    def __init_transform(self):
        transform_list = [transforms.Resize(224),
                          transforms.RandomCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])]
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
        np.savez('./result/{}_{}.npz'.format(self.args.weight_dir, name), array)
