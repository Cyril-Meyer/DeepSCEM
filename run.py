import argparse
import sys

from PyQt5 import QtWidgets

from mainwindow import MainWindow

import data

from manager import Manager

parser = argparse.ArgumentParser()
parser.add_argument('--create-dataset', type=str, action='append', nargs=3,
                    help='create a dataset.'
                         '<filename> <name> <n_labels>')
parser.add_argument('--add-sample', type=str, action='append', nargs='+',
                    help='add a sample from image and labels to an existing dataset.\n'
                         '<filename> <name> <image> [<label1> <label2> ...]')
args = parser.parse_args()

# create dataset module
if args.create_dataset:
    print(args.create_dataset)
    for dataset in args.create_dataset:
        dataset_parser = argparse.ArgumentParser()
        dataset_parser.add_argument('filename', type=str)
        dataset_parser.add_argument('name', type=str)
        dataset_parser.add_argument('labels', type=int)
        dataset_args = dataset_parser.parse_args(dataset)

        manager = Manager()
        manager.new_dataset(dataset_args.name, dataset_args.labels, dataset_args.filename)

# add sample module
if args.add_sample:
    for sample in args.add_sample:
        sample_parser = argparse.ArgumentParser()
        sample_parser.add_argument('filename', type=str)
        sample_parser.add_argument('name', type=str)
        sample_parser.add_argument('image', type=str)
        sample_parser.add_argument('labels', type=str, nargs='*')
        sample_args = sample_parser.parse_args(sample)

        manager = Manager()
        dataset = manager.load_dataset(sample_args.filename)
        manager.add_sample(dataset, sample_args.name, sample_args.image, sample_args.labels)

# no args = graphical mode
if not len(sys.argv) > 1:
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()
