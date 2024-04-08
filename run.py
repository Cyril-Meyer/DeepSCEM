import argparse
import sys

from PyQt5 import QtWidgets

from mainwindow import MainWindow

import data

parser = argparse.ArgumentParser()
parser.add_argument('--create-sample', type=str, action='append', nargs='+',
                    help='create a sample from image and labels.\n<name> <image> [<label1> <label2> ...]')
parser.add_argument('--create-dataset', type=str, action='append', nargs='+',
                    help='create a dataset from samples.\n <name> <sample1> [<sample2> <sample3> ...]')
args = parser.parse_args()

# create sample module
if args.create_sample:
    for sample in args.create_sample:
        sample_parser = argparse.ArgumentParser()
        sample_parser.add_argument('name', type=str)
        sample_parser.add_argument('image', type=str)
        sample_parser.add_argument('labels', type=str, nargs='*')
        sample_args = sample_parser.parse_args(sample)

        data.create_sample(sample_args.name, sample_args.image, sample_args.labels)

# create dataset module
if args.create_dataset:
    for dataset in args.create_dataset:
        dataset_parser = argparse.ArgumentParser()
        dataset_parser.add_argument('name', type=str)
        dataset_parser.add_argument('samples', type=str, nargs='*')
        dataset_args = dataset_parser.parse_args(dataset)

        data.create_dataset(dataset_args.name, dataset_args.samples)

# no args = graphical mode
if not len(sys.argv) > 1:
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()
