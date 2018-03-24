from trainer import Trainer
import argparse
import torch

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Chest X-ray classification CNN in PyTorch')
    parser.add_argument('--model', type=str, default='DenseNet',
                        help='The model name [DenseNet121, DenseNet161, DenseNet169, '
                             'DenseNet201, CheXNet, ResNet18, ResNet34, ResNet50,'
                             ' ResNet101, ResNet152, VGG191]')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Using pretrained or not')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='the path of the data directory')
    parser.add_argument('--train-csv', type=str, default='./data',
                        help='the path of the train label csv directory')
    parser.add_argument('--val-csv', type=str, default='../data',
                        help='the path of the val label csv directory')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--reshape-size', type=int, default=256,
                        help='the size after reshaping')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='the size after cropping')
    parser.add_argument('--weight-dir', type=str, default=None,
                        help='the path of the model if keep training')
    parser.add_argument('--classes', type=int, default=14,
                        help='the #classes of target')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    trainer = Trainer(args)
    trainer.train()
