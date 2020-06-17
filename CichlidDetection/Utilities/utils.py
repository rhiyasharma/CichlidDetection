import os, subprocess


def make_dir(path):
    """recursively create the directory specified by path if it does not exist

    Args:
        path: path to the directory that will be created

    Returns:
        str: path, identical to the input argument path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    assert os.path.exists(path), "failed to create {}".format(path)
    return path


def run(command, fault_tolerant=False):
    """use the subprocess.run function to run a command

    Args:
        command: list of strings to be passed as the first argument of subprocess.run()
        fault_tolerant: if False (default) and the command fails to run, raise an exception and halt the script

    Returns:
        str: stdout from executing subprocess.run(command)

    Raises:
        Exception: if subprocess.run() produces a nonzero return code and fault_tolerant is False
    """

    output = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8')
    if output.returncode != 0:
        if not fault_tolerant:
            print(output.stderr)
            raise Exception('error running the following command: {}'.format(' '.join(command)))
        else:
            print('error running the following command: {}'.format(' '.join(command)))
            print('fault tolerant set to True, ignoring error')
    return output.stdout


def xyminmax_to_xywh(xmin, ymin, xmax, ymax):
    """convert box coordinates from (xmin, ymin, xmax, ymax) form to (x, y, w , h) form"""
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def xywh_to_xyminmax(x, y, w, h):
    """convert box coordinates from (x, y, w , h) form to (xmin, ymin, xmax, ymax) form"""
    return [x, y, x + w, y + h]
