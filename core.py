from CichlidDetection.Classes.Runner import Runner
import argparse, subprocess, os

# primary command line executable script

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')
download_parser = subparsers.add_parser('download')
train_parser = subparsers.add_parser('train')

train_parser.add_argument('-e', '--Epochs', type=int, default=10)
train_parser.add_argument('--pbs', action='store_true')
args = parser.parse_args()

runner = Runner()
package_root = os.path.dirname(os.path.abspath(__file__))

if args.command == 'download':
    runner.download()

elif args.command == 'train':
    if args.pbs:
        pbs_dir = os.path.join(package_root, 'CichlidDetection/PBS')
        subprocess.run(['cd {}; qsub train.pbs'.format(pbs_dir)])
    else:
        runner.prep()
        runner.train(num_epochs=args.Epochs)


