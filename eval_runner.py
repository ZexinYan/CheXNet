from tester import ChexnetTester
import argparse
import torch


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='PyTorch To test Chest-Xray by using densenet')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='the path of the data directory')
    parser.add_argument('--test-csv', type=str, default='',
                        help='the path of the test csv')
    parser.add_argument('--model-dir', type=str, default='',
                        help='the path of trained model')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='the batch size when testing')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--reshape-size', type=int, default=224,
                        help='the size of the input image')
    parser.add_argument('--classes', type=int, default=14,
                        help='the #classes of target')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    tester = ChexnetTester(args=args)
    tester.test()
