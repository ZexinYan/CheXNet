from trainer import ChexnetTrainer
import argparse
import torch


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Chest-Xray by using densenet')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='the path of the data directory')
    parser.add_argument('--train-csv', type=str, default='../data',
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
    parser.add_argument('--model-dir', type=str, default=None,
                        help='the path of the model if keep training')
    parser.add_argument('--saved-dir', type=str, default=None,
                        help='the path of the saved model')
    parser.add_argument('--show-model', action='store_true', default=False,
                        help='show the model summary')
    parser.add_argument('--classes', type=int, default=14,
                        help='the #classes of target')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    trainer = ChexnetTrainer(args)

    trainer.train()
