import argparse
from CichlidDetection.Classes.DataPrepper import DataPrepper

# main script, meant to be run from the command line.

dataprepper = DataPrepper('MC6_5')
dataprepper.prep()
