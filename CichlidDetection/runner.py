import argparse, os, sys
from CichlidDetection.Classes.DataPrepper import DataPrepper

# main script. Can be run from the command line by navigating outermost CichlidDetection folder and
# calling 'python -m CichlidDetection.runner'

dataprepper = DataPrepper('MC6_5')
dataprepper.prep()
