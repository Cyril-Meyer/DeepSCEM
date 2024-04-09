import time

import manager

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from mainwindowui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.manager = manager.Manager()

    def dataset_update(self):
        self.treeWidget_dataset.clear()
        for dataset in self.manager.get_dataset_index():
            dataset_item = QTreeWidgetItem(self.treeWidget_dataset, [dataset])
            # todo: make it work using manager
            '''
            for data in dataset:
                QTreeWidgetItem(dataset_item, [data])
            '''

            self.treeWidget_dataset.addTopLevelItem(dataset_item)

    def dataset_load_clicked(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Select datasets', '', 'Datasets (*.h5 *.hdf5)')
        try:
            for filename in filenames:
                self.manager.load_dataset(filename)
            self.dataset_update()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Dataset load error.\n{e}')

    def dataset_unload_clicked(self):
        return

    def dataset_saveas_clicked(self):
        return

    def sample_add_clicked(self):
        self.sample_add_wizard()

    def sample_remove_clicked(self):
        return

    def sample_add_wizard(self):
        # Select destination dataset or create a new one
        dataset_list = ['Create new dataset'] + self.manager.get_dataset_index()
        choice_dataset, ok = QInputDialog.getItem(self, 'Add sample', 'Add sample to Dataset', dataset_list, editable=False)
        if not ok:
            return
        if choice_dataset == 'Create new dataset':
            choice_dataset, ok = QInputDialog.getText(self, 'Add sample',
                                                      'Dataset name (empty will create a name for you)')
            if not ok:
                return
            if len(choice_dataset) < 1:
                choice_dataset = f'{time.time():.4f}'.replace('.', '_')

        # Add new sample
        choice_sample, ok = QInputDialog.getText(self, 'Add sample', 'Sample name')
        if not ok:
            return
        if len(choice_sample) < 1:
            choice_sample = f'{time.time():.4f}'.replace('.', '_')
        # Select sample image
        choice_image, _ = QFileDialog.getOpenFileName(self, 'Select sample image', '', 'Sample (*.tif *.tiff *.npy)')
        if choice_image == '':
            return
        # Select sample labels
        choice_number_label, ok = QInputDialog.getInt(self, 'Sample labels', 'Number of labels for sample', 1, 0, 1000)
        if not ok:
            return
        choice_labels = []
        for i in range(choice_number_label):
            choice_label, _ = QFileDialog.getOpenFileName(self, f'Select sample label {i}. '
                                                                f'No file will be considered a blank label.',
                                                          '', 'Sample (*.tif *.tiff *.npy)')
            choice_labels.append(choice_label)

        # Create everything for real
        try:
            self.manager.new_dataset(choice_dataset)
            self.manager.add_sample(choice_dataset, choice_sample, choice_image, choice_labels)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'{e}')
            self.manager.remove_dataset(choice_dataset)

        # Update view
        self.dataset_update()
