import argparse, os, sys
from CichlidDetection.Classes.DataPrepper import DataPrepper
import pdb

# main script. Can be run from the command line by navigating outermost CichlidDetection folder and
# calling 'python -m CichlidDetection.runner'
pdb.set_trace()
dataprepper = DataPrepper('MC6_5')
dataprepper.YOLO_prep()
