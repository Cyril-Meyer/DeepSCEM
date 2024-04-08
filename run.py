import argparse
import sys

from PyQt5 import QtWidgets

from mainwindow import MainWindow

parser = argparse.ArgumentParser()
parser.add_argument('--create-sample', type=str, action='append', nargs='+',
                    help='create a sample from image and labels.\n<name> <image> [<label1> <label2> ...]')
parser.add_argument('--create-dataset', type=str, action='append', nargs='+',
                    help='create a dataset from samples.\n <name> <sample1> [<sample2> <sample3> ...]')
args = parser.parse_args()

# no args = graphical mode
if not len(sys.argv) > 1:
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()
