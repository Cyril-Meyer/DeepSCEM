from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from mainwindowui import Ui_MainWindow

import manager


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.manager = manager.Manager()

    def dataset_update(self):
        self.treeWidget_dataset.clear()
        for dataset in self.manager.datasets:
            dataset_item = QTreeWidgetItem(self.treeWidget_dataset, [dataset.filename])
            for data in dataset:
                QTreeWidgetItem(dataset_item, [data])

            self.treeWidget_dataset.addTopLevelItem(dataset_item)

    def dataset_load_clicked(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Select datasets', '', 'Datasets (*.h5, *.hdf5)')
        try:
            for filename in filenames:
                self.manager.load_dataset(filename)
            self.dataset_update()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Dataset load error.\n{e}')
