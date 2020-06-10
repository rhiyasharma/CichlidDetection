from .Classes.Runner import Runner
import argparse

# primary command line executable script

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')
download_parser = subparsers.add_parser('download')
train_parser = subparsers.add_parser('train')

train_parser.add_argument('-e', '--Epochs', type=int, default=10)
args = parser.parse_args()

runner = Runner()

if args.command == 'download':
    runner.download()

elif args.command == 'train':
    runner.prep()
    runner.train(num_epochs=args.Epochs)


