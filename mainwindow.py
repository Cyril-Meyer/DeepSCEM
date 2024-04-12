import time
import numpy as np

import manager

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from mainwindowui import Ui_MainWindow
from customDialogs import DialogNewModel


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.mainLayout.setStretch(0, 0)
        self.mainLayout.setStretch(1, 4)
        self.manager = manager.Manager()
        self.view_selection = None
        self.flag_disable_ui_events = False

    # ----------------------------------------
    # Debug call (future "about" messagebox)
    # ----------------------------------------
    def menu_help_about_triggered(self, event):
        self.dataset_item_selection_changed()

    # ----------------------------------------
    # Window events
    # ----------------------------------------
    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        self.mainview_update()

    # ----------------------------------------
    # Widgets update
    # ----------------------------------------
    def mainview_update(self):
        """
        Update the mainview (image and labels view)

        The first selected element is the most important element.
        Others elements are draw over.
        """
        self.label_mainview.clear()
        if self.view_selection is None:
            return
        data, selection = self.view_selection
        if len(selection) <= 0:
            return
        # todo: add a check to know if redraw is needed if nothing changed

        # Get data with z selection
        data_view = data[selection[0]]
        self.horizontalSlider_z.setMaximum(data_view.shape[0]-1)
        z = self.horizontalSlider_z.value()
        data_view = data_view[z]

        # data convert [0, 1] -> [0, 255]
        # data_view = (np.clip(data_view, 0, 1) * 255).astype(np.uint8)
        data_view = (data_view * 255).astype(np.uint8)
        height, width = data_view.shape

        # grayscale if only one image, RGB otherwise
        if len(selection) == 1:
            bytesPerLine = 1 * width
            data_view_qimage = QImage(data_view.data, data_view.shape[1], data_view.shape[0],
                                      bytesPerLine, QImage.Format_Grayscale8)
        else:
            data_view = np.stack((data_view,) * 3, axis=-1)
            for i, s in enumerate(selection[1:]):
                data_view[data[s][z] > 0, i % 3] = 255

            bytesPerLine = 3 * width
            data_view_qimage = QImage(data_view.data, data_view.shape[1], data_view.shape[0],
                                      bytesPerLine, QImage.Format_RGB888)

        qImg = QPixmap(data_view_qimage).scaled(self.label_mainview.width()-50,
                                                self.label_mainview.height()-50,
                                                Qt.KeepAspectRatio)
        self.label_mainview.setPixmap(qImg)

    def dataset_update(self):
        self.flag_disable_ui_events = True
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
        self.flag_disable_ui_events = False

    # ----------------------------------------
    # Widgets information processing (input)
    # ----------------------------------------
    def dataset_get_selection_hierarchy(self):
        current_item = self.treeWidget_dataset.currentItem()
        texts = []
        indexes = []
        items = []
        while current_item is not None:
            items.append(current_item)
            texts.append(current_item.text(0))
            indexes.append(self.treeWidget_dataset.indexOfTopLevelItem(current_item))
            current_item = current_item.parent()
        return texts, indexes, items

    def dataset_get_selection(self):
        # todo: refactor: use dataset_get_selection_hierarchy
        current_item = self.treeWidget_dataset.currentItem()
        text = None
        index = -1
        while current_item is not None:
            text = current_item.text(0)
            index = self.treeWidget_dataset.indexOfTopLevelItem(current_item)
            current_item = current_item.parent()
        return text, index

    # ----------------------------------------
    # User events on ui
    # ----------------------------------------
    def mainview_mouse_event(self, event):
        return

    def mainview_slider_changed(self):
        self.mainview_update()

    def dataset_item_selection_changed(self):
        if self.flag_disable_ui_events:
            return
        _, _, selection = self.dataset_get_selection_hierarchy()
        # nothing selected
        if len(selection) < 1:
            self.view_selection = None
        # dataset selected
        elif len(selection) == 1:
            # print(selection[0].child(0).text(0))
            self.view_selection = None
        # sample selected
        else:
            sample = selection[-2]
            dataset_name = selection[-1].text(0)
            sample_name = sample.text(0)
            sample_data = self.manager.get_sample(dataset_name, sample_name)
            sample_view = []

            for i in range(sample.childCount()):
                if sample.child(i).checkState(0) != 0:
                    sample_view.append(sample.child(i).text(0))

            self.view_selection = (sample_data, sample_view)
        self.mainview_update()

    # ----------------------------------------
    # User events on buttons and menus
    # ----------------------------------------
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
            self.dataset_update()

    def dataset_unload_clicked(self):
        text, index = self.dataset_get_selection()
        if index >= 0 and text is not None:
            self.manager.remove_dataset(text)
            self.dataset_update()
        else:
            QMessageBox.information(self, 'Warning', f'No dataset selected.')

    def dataset_saveas_clicked(self):
        text, index = self.dataset_get_selection()
        if index >= 0 and text is not None:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save dataset', '', 'Datasets (*.h5 *.hdf5)')
            if not filename == '':
                self.manager.saveas_dataset(text, filename)
        else:
            QMessageBox.information(self, 'Warning', f'No dataset selected.')

    def sample_add_clicked(self):
        self.sample_add_wizard()
        self.dataset_update()

    def sample_remove_clicked(self):
        texts, indexes, items = self.dataset_get_selection_hierarchy()
        if len(indexes) < 2:
            QMessageBox.information(self, 'Warning', f'No sample selected.')
        else:
            try:
                self.manager.remove_sample(dataset=texts[-1], sample=texts[-2])
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'{e}')
        self.dataset_update()

    def model_load_clicked(self):
        '''
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Select models', '', 'Model (*.h5)')
        try:
            for filename in filenames:
                self.manager.load_model(filename)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Dataset load error.\n{e}')
        '''
        return

    def model_new_clicked(self):
        dialog = DialogNewModel()
        if dialog.exec() == 1:
            dim, archi, block, depth, outputs, acti = dialog.get()

    def model_train_clicked(self):
        return

    def model_pred_clicked(self):
        return

    # ----------------------------------------
    # Wizards (multi-step user input)
    # ----------------------------------------
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
                self.dataset_new_manager(choice_dataset, choice_number_label)
            self.sample_add_manager(choice_dataset, choice_sample, choice_image, choice_labels)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'{e}')
            self.manager.remove_dataset(choice_dataset)

    # ----------------------------------------
    # Wizards data manager
    # ----------------------------------------
    def sample_add_manager(self, choice_dataset, choice_sample, choice_image, choice_labels):
        self.manager.add_sample(choice_dataset, choice_sample, choice_image, choice_labels)

    def dataset_new_manager(self, choice_dataset, choice_number_label):
        self.manager.new_dataset(choice_dataset, choice_number_label)
