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
parser.add_argument('--train-model', type=str, nargs='+',
                    help='train a model.\n'
                         '<model input filename> <model output filename>'
                         '<train dataset filename> <valid dataset filename> '
                         '<loss> <batch_size>'
                         '<patch_size_z> <patch_size_y> <patch_size_x>'
                         '<steps_per_epoch> <epochs> <validation_steps>')
parser.add_argument('--gui-no-safe', action='store_true')
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

# train model
if args.train_model:
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument('model_input', type=str)
    train_parser.add_argument('model_output', type=str)
    train_parser.add_argument('train', type=str)
    train_parser.add_argument('valid', type=str)
    train_parser.add_argument('loss', type=str)
    train_parser.add_argument('batch_size', type=int)
    train_parser.add_argument('patch_size_z', type=int)
    train_parser.add_argument('patch_size_y', type=int)
    train_parser.add_argument('patch_size_x', type=int)
    train_parser.add_argument('steps_per_epoch', type=int)
    train_parser.add_argument('epochs', type=int)
    train_parser.add_argument('validation_steps', type=int)
    train_args = train_parser.parse_args(args.train_model)

    manager = Manager()
    model = manager.load_model(train_args.model_input)
    train = manager.load_dataset(train_args.train)
    try:
        valid = manager.load_dataset(train_args.valid)
    except ValueError:
        valid = train

    manager.train_model(0, train, valid,
                        train_args.loss, train_args.batch_size,
                        train_args.patch_size_z, train_args.patch_size_y, train_args.patch_size_x,
                        train_args.steps_per_epoch, train_args.epochs, train_args.validation_steps)
    manager.save_model(0, train_args.model_output)

# no args = graphical mode
if not len(sys.argv) > 1 or args.gui_no_safe:
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow(safe=(not args.gui_no_safe))
    window.show()
    app.exec()
