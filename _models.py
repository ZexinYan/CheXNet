import torch
import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        num_in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_in_features, classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class DenseNet161(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet161, self).__init__()
        self.model = torchvision.models.densenet161(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DenseNet169(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet169, self).__init__()
        self.model = torchvision.models.densenet169(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet201, self).__init__()
        self.model = torchvision.models.densenet201(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class CheXNet(nn.Module):
    def __init__(self, classes=156):
        super(CheXNet, self).__init__()
        self.densenet121 = DenseNet121(classes=14)
        self.densenet121 = torch.nn.DataParallel(self.densenet121).cuda()
        self.densenet121.load_state_dict(torch.load('./models/CheXNet.pth.tar')['state_dict'])
        self.densenet121.module.densenet121.classifier = nn.Sequential(
            nn.Linear(1024, classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet101, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet152(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet152, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class VGG19(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(VGG19, self).__init__()
        self.model = torchvision.models.vgg19_bn(pretrained=pretrained)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(InceptionV3, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_in_features, classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    classes = 156
    model = DenseNet121(classes=classes, pretrained=True)
    model = DenseNet161(classes=classes, pretrained=True)
    model = DenseNet169(classes=classes, pretrained=True)
    model = DenseNet201(classes=classes, pretrained=True)
    model = ResNet18(classes=classes, pretrained=True)
    model = ResNet34(classes=classes, pretrained=True)
    model = ResNet50(classes=classes, pretrained=True)
    model = ResNet101(classes=classes, pretrained=True)
    model = ResNet152(classes=classes, pretrained=True)
    model = VGG19(classes=classes, pretrained=True)
    model = CheXNet(classes=classes)
    # model = InceptionV3(classes=classes, pretrained=True)
