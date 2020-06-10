import argparse, subprocess, os

# primary command line executable script

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')

download_parser = subparsers.add_parser('download')

train_parser = subparsers.add_parser('train')
train_parser.add_argument('-e', '--Epochs', type=int, default=10)

full_auto_parser = subparsers.add_parser('full_auto')
full_auto_parser.add_argument('-e', '--Epochs', type=int, default=10)
full_auto_parser.add_argument('--pbs', action='store_true')

args = parser.parse_args()

package_root = os.path.dirname(os.path.abspath(__file__))

if args.command == 'full_auto' and args.pbs:
    pbs_dir = os.path.join(package_root, 'CichlidDetection/PBS')
    subprocess.run(['qsub', 'train.pbs' '-v' 'EPOCHS="{}"'.format(args.Epochs)], cwd=pbs_dir)

else:
    from CichlidDetection.Classes.Runner import Runner
    runner = Runner()

    if args.command == 'full_auto':
        runner.download()
        runner.prep()
        runner.train(num_epochs=args.Epochs)

    elif args.command == 'download':
        runner.download()

    elif args.command == 'train':
        runner.prep()
        runner.train(num_epochs=args.Epochs)




