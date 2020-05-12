import os, subprocess


def make_dir(path):
    """recursively create the directory specified by path if it does not exist
    :param path: path to the directory that will be created
    """
    if not os.path.exists(path):
        os.makedirs(path)
    assert os.path.exists(path), "failed to create {}".format(path)


def run(command, fault_tolerant=False):
    """use the subprocess.run function to run a command
    :param command: list of strings to be passed as the first argument of subprocess.run()
    :param fault_tolerant: if False (default) and the command fails to run, raise an exception and halt the script
    :return: output of subprocess.run() command
    """
    output = subprocess.run(command, capture_output=True, encoding='utf-8')
    if output.returncode != 0:
        if not fault_tolerant:
            print(output.stderr)
            raise Exception('error running the following command: {}'.format(' '.join(command)))
        else:
            print('error running the following command: {}'.format(' '.join(command)))
            print('fault tolerant set to True, ignoring error')
    return output.stdout
