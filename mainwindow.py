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

    def mainview_mouse_event(self, event):
        print(event)
        return

    def dataset_get_selection(self):
        current_item = self.treeWidget_dataset.currentItem()
        text = None
        index = -1
        while current_item is not None:
            text = current_item.text(0)
            index = self.treeWidget_dataset.indexOfTopLevelItem(current_item)
            current_item = current_item.parent()
        return text, index

    def dataset_update(self):
        self.treeWidget_dataset.clear()
        for dataset in self.manager.get_datasets_index():
            dataset_item = QTreeWidgetItem(self.treeWidget_dataset, [dataset])
            for sample, samples_data in self.manager.get_dataset_samples(dataset):
                sample_item = QTreeWidgetItem(dataset_item, [sample])
                for i, sample_data in enumerate(samples_data):
                    item = QTreeWidgetItem(sample_item, [sample_data])
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, i == 0)

            self.treeWidget_dataset.addTopLevelItem(dataset_item)
        self.treeWidget_dataset.expandAll()

    def dataset_new_clicked(self):
        choice_dataset, choice_number_label, ok = self.dataset_new_wizard()
        if ok:
            self.dataset_new_manager(choice_dataset, choice_number_label)
            self.dataset_update()

    def dataset_load_clicked(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Select datasets', '', 'Datasets (*.h5 *.hdf5)')
        try:
            for filename in filenames:
                self.manager.load_dataset(filename)
            self.dataset_update()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Dataset load error.\n{e}')

    def dataset_unload_clicked(self):
        self.dataset_update()

    def dataset_saveas_clicked(self):
        self.dataset_get_selection()
        return

    def sample_add_clicked(self):
        self.sample_add_wizard()

    def sample_remove_clicked(self):
        return

    def dataset_new_wizard(self):
        choice_dataset, ok = QInputDialog.getText(self, 'New Dataset',
                                                  'Dataset name (empty will create a name for you)')
        if not ok:
            return None, None, ok
        if len(choice_dataset) < 1:
            choice_dataset = f'{time.time():.4f}'.replace('.', '_')

        choice_number_label, ok = QInputDialog.getInt(self, 'New Dataset', 'Number of labels per sample', 1, 0, 1000)

        if not ok:
            return None, None, ok

        return choice_dataset, choice_number_label, ok

    def sample_add_wizard(self):
        # Select destination dataset or create a new one
        dataset_list = ['Create new dataset'] + self.manager.get_datasets_index()
        text, index = self.dataset_get_selection()
        choice_dataset, ok = QInputDialog.getItem(self, 'Add sample', 'Add sample to Dataset', dataset_list,
                                                  current=index+1, editable=False)
        if not ok:
            return
        new_dataset = False
        if choice_dataset == 'Create new dataset':
            new_dataset = True
            choice_dataset, choice_number_label, ok = self.dataset_new_wizard()
            if not ok:
                return
        else:
            choice_number_label = self.manager.get_datasets_number_labels(choice_dataset)

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
        choice_labels = []
        for i in range(choice_number_label):
            choice_label, _ = QFileDialog.getOpenFileName(self, f'Select sample label {i}. '
                                                                f'No file will be considered a blank label.',
                                                          '', 'Sample (*.tif *.tiff *.npy)')
            choice_labels.append(choice_label)

        # Create everything for real
        try:
            if new_dataset:
                self.manager.new_dataset(choice_dataset, choice_number_label)
            self.manager.add_sample(choice_dataset, choice_sample, choice_image, choice_labels)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'{e}')
            self.manager.remove_dataset(choice_dataset)

    def sample_add_manager(self):
        # todo: put code at the end of the wizard here
        return

    def dataset_new_manager(self, choice_dataset, choice_number_label):
        self.manager.new_dataset(choice_dataset, choice_number_label)
