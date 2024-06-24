import time
import numpy as np

import manager

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from mainwindowui import Ui_MainWindow
from customDialogs import DialogNewModel, DialogTrain, DialogPred, DialogEval, DialogEvalRes


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, safe=True, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.mainLayout.setStretch(0, 0)
        self.mainLayout.setStretch(1, 4)
        self.manager = manager.Manager()
        self.view_selection = None
        self.flag_disable_ui_events = False
        # Blocking task: run a blocking task in a thread but keep ui in a "disable" state
        self.bt_thread = None
        self.bt_worker = None
        self.bt_messagebox = None
        # Safe Mode
        self.safe = safe
        self.safe_labels = None

    # ----------------------------------------
    # About
    # ----------------------------------------
    def menu_help_about_triggered(self):
        QMessageBox.about(self, 'DeepSCEM', "Cyril Meyer 2024<br>"
                                            "Deep Segmentation for Cellular Electron Microscopy<br><br>"
                                            "<a href='https://github.com/Cyril-Meyer/DeepSCEM'>Cyril-Meyer/DeepSCEM</a>")

    def menu_help_about_qt_triggered(self):
        QMessageBox.aboutQt(self, 'Qt')

    # ----------------------------------------
    # Safe mode
    # ----------------------------------------
    def safe_mode_disable(self):
        if self.safe is False:
            return
        self.safe = False
        self.actionDistance_transform.setEnabled(True)
        QMessageBox.information(self, 'Safe mode disabled', 'Safe mode disabled.')

    def safe_mode_get_labels(self):
        if self.safe:
            if self.safe_labels is None:
                self.safe_mode_set_labels()
            self.statusbar.showMessage(f'Safe mode enabled. DeepSCEM is locked in {self.safe_labels} '
                                       f'{"classes" if self.safe_labels > 1 else "class"} mode.')
            return self.safe_labels
        else:
            return None

    def safe_mode_set_labels(self, labels=None):
        if labels is None:
            ok = False
            while not ok:
                choice_number_label, ok = QInputDialog.getInt(self, 'Safe mode', 'Number of labels', 1, 0, 1000)

            self.safe_labels = choice_number_label
        else:
            labels = max(0, int(labels))
            self.safe_labels = labels

    def safe_mode_warning(self):
        QMessageBox.information(self, 'Safe mode', 'This feature is disabled because because of safe mode.\n'
                                                   'DeepSCEM is currently running in safe mode.\n'
                                                   'To disable safe mode, use the "Help" menu and click '
                                                   '"Turn off safe mode".')

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

    def screenshot(self):
        img = self.label_mainview.pixmap()
        if img is not None:
            if img.save(f'{time.time():.4f}'.replace('.', '_') + '.png'):
                return

        QMessageBox.information(self, 'Warning', f'Nothing to screenshot.')

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

    def models_update(self):
        self.flag_disable_ui_events = True
        self.listWidget_model.clear()
        for model in self.manager.get_models_list():
            self.listWidget_model.addItem(model)
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

    def dataset_item_rename(self):
        _, _, selection = self.dataset_get_selection_hierarchy()
        # nothing selected
        if len(selection) < 1:
            QMessageBox.information(self, 'Warning', f'No dataset or sample selected.')
            return
        # dataset
        elif len(selection) == 1:
            dataset_name = selection[0].text(0)

            new_name, ok = QInputDialog.getText(self, 'New name', 'New dataset name', text=dataset_name)
            if not ok or len(new_name) <= 0:
                return
            if new_name in self.manager.get_datasets_index():
                QMessageBox.critical(self, 'Warning', f'Name already exist.')
                return

            self.manager.rename_dataset(dataset_name, new_name)
            self.dataset_update()
        # sample
        else:
            dataset_name = selection[-1].text(0)
            sample_name = selection[-2].text(0)

            new_name, ok = QInputDialog.getText(self, 'New name', 'New sample name', text=sample_name)
            if not ok or len(new_name) <= 0:
                return
            if new_name in self.manager.get_dataset_samples(dataset_name, info=False):
                QMessageBox.critical(self, 'Warning', f'Name already exist.')
                return

            self.manager.rename_sample(dataset_name, sample_name, new_name)
            self.dataset_update()

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

    def dataset_load(self, filename):
        labels = None
        if self.safe:
            labels = self.safe_labels

        try:
            name = self.manager.load_dataset(filename, labels)
        except AssertionError as e:
            QMessageBox.critical(self, 'Error', f'Safe mode dataset load error. '
                                                f'Disable safe mode to load datasets with different number of classes. '
                                                f'\n{e}')
            return
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Dataset load error.\n{e}')
            return

        if self.safe and self.safe_labels is None:
            labels = self.manager.get_datasets_number_labels(name)
            self.safe_labels = labels

    def dataset_load_clicked(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Select datasets', '', 'Datasets (*.hdf5)')

        for filename in filenames:
            self.dataset_load(filename)
        self.dataset_update()

    def dataset_load_drop(self, url):
        file = url.toLocalFile()
        self.dataset_load(file)
        self.dataset_update()

    def dataset_unload_clicked(self):
        text, index = self.dataset_get_selection()
        if index >= 0 and text is not None:
            self.manager.remove_dataset(text)
            self.dataset_update()
            self.view_selection = None
        else:
            QMessageBox.information(self, 'Warning', f'No dataset selected.')

    def dataset_saveas_clicked(self):
        text, index = self.dataset_get_selection()
        if index >= 0 and text is not None:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save dataset', '', 'Datasets (*.hdf5)')
            if not filename == '':
                if not (filename.endswith('.h5') or filename.endswith('.hdf5')):
                    filename += '.hdf5'
                self.manager.saveas_dataset(text, filename)
        else:
            QMessageBox.information(self, 'Warning', f'No dataset selected.')

    def dataset_distance_transform(self):
        if self.safe:
            self.safe_mode_warning()
            return

        text, index = self.dataset_get_selection()
        if index >= 0 and text is not None:
            self.manager.distance_transform(text)
        else:
            QMessageBox.information(self, 'Warning', f'No dataset selected.')
        self.dataset_update()

    def sample_add_clicked(self):
        self.sample_add_wizard()
        self.dataset_update()

    def sample_crop_clicked(self):
        texts, indexes, items = self.dataset_get_selection_hierarchy()
        if len(indexes) < 2:
            QMessageBox.information(self, 'Warning', f'No sample selected.')
        else:
            try:
                self.sample_crop_wizard(dataset=texts[-1], sample=texts[-2])
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'{e}')
        self.dataset_update()

    def sample_remove_clicked(self):
        texts, indexes, items = self.dataset_get_selection_hierarchy()
        if len(indexes) < 2:
            QMessageBox.information(self, 'Warning', f'No sample selected.')
        else:
            try:
                self.manager.remove_sample(dataset=texts[-1], sample=texts[-2])
                self.view_selection = None
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'{e}')
        self.dataset_update()

    def model_load_clicked(self):
        labels = self.safe_mode_get_labels()
        filename, _ = QFileDialog.getOpenFileName(self, 'Select model', '', 'Model (*.h5)')
        if filename is None or filename == '':
            return
        try:
            if self.safe:
                self.manager.load_model(filename, labels)
                self.models_update()
            else:
                self.blocking_task(target=self.manager.load_model,
                                   args=(filename,),
                                   message='Loading model...',
                                   target_end=self.models_update,
                                   wait_end=False)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Model load error.\n{e}')
        return

    def model_new_clicked(self):
        labels = self.safe_labels if self.safe else None
        dialog = DialogNewModel(labels, self)
        if dialog.exec() == 1:
            a0, a1, a2, a3, a4, a5, a6, a7, outputs, a9 = dialog.get()
            self.blocking_task(target=self.manager.new_model,
                               args=(a0, a1, a2, a3, a4, a5, a6, a7, outputs, a9),
                               message='Creating model...',
                               target_end=self.models_update,
                               wait_end=False)
            # self.manager.new_model(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)
            # self.models_update()
            if self.safe and self.safe_labels is None:
                self.safe_labels = outputs

    def model_save_clicked(self):
        index = self.listWidget_model.currentRow()
        if index >= 0:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save model', '', 'Model (*.h5)')
            if not filename == '':
                if not filename.endswith('.h5'):
                    filename += '.h5'
                self.manager.save_model(index, filename)
        else:
            QMessageBox.information(self, 'Warning', f'No model selected.')

    # ----------------------------------------
    # User events forms
    # ----------------------------------------
    def model_train_clicked(self):
        if len(self.manager.get_datasets_index()) <= 0:
            QMessageBox.information(self, 'Warning', f'No dataset to select.')
            return
        if len(self.manager.get_models_list()) <= 0:
            QMessageBox.information(self, 'Warning', f'No model to select.')
            return

        dialog = DialogTrain(self.manager.get_datasets_index(), self.manager.get_models_list(), self)
        if dialog.exec() == 1:
            model_index, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, augmentations, focus = dialog.get()
            self.blocking_task(target=self.manager.train_model,
                               args=(model_index, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, augmentations, focus),
                               message='Training model...')
            # self.manager.train_model(model_index, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)

    def model_pred_clicked(self):
        if len(self.manager.get_datasets_index()) <= 0:
            QMessageBox.information(self, 'Warning', f'No dataset to select.')
            return
        if len(self.manager.get_models_list()) <= 0:
            QMessageBox.information(self, 'Warning', f'No model to select.')
            return

        dialog = DialogPred(self.manager.get_datasets_index(), self.manager.get_models_list(), self)
        if dialog.exec() == 1:
            model_index, a0, a1, a2, a3, a4, thres, thres_val = dialog.get()
            self.blocking_task(target=self.manager.pred_model,
                               args=(model_index, a0, a1, a2, a3, a4,
                                     thres_val if thres else None),
                               message='Predicting...',
                               target_end=self.dataset_update)

    def model_eval_clicked(self):
        if self.safe:
            self.safe_mode_warning()
            return

        if len(self.manager.get_datasets_index()) <= 0:
            QMessageBox.information(self, 'Warning', f'No dataset to select.')
            return
        # Evaluation dialog
        dialog = DialogEval(self.manager.get_datasets_index(), self)
        if dialog.exec() == 1:
            ref, seg, f1, iou = dialog.get()

            try:
                result = self.manager.eval_dataset(ref, seg, f1, iou)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'{e}')
                return

            # Evaluation results dialog
            dialog = DialogEvalRes(result, self)
            dialog.exec()

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

        if self.safe:
            choice_number_label = self.safe_mode_get_labels()
            ok = True
        else:
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
        choice_image, _ = QFileDialog.getOpenFileName(self, 'Select sample image', '', 'Sample (*.tif *.tiff)')
        if choice_image == '':
            return
        # Select sample labels
        choice_labels = []
        for i in range(choice_number_label):
            choice_label, _ = QFileDialog.getOpenFileName(self, f'Select sample label {i}. '
                                                                f'No file will be considered a blank label.',
                                                          '', 'Sample (*.tif *.tiff)')
            choice_labels.append(choice_label)

        # Create everything for real
        try:
            if new_dataset:
                self.dataset_new_manager(choice_dataset, choice_number_label)
            self.sample_add_manager(choice_dataset, choice_sample, choice_image, choice_labels)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'{e}')
            self.manager.remove_dataset(choice_dataset)

    def sample_crop_wizard(self, dataset, sample):
        z_max, y_max, x_max = tuple(self.manager.get_sample(dataset, sample).attrs['shape'])

        def get_min_max(v_max, axis):
            v_min, ok1 = QInputDialog.getInt(self, 'Crop sample', f'{axis} min', value=0, min=0, max=v_max-1)
            if not ok1:
                return None, None, ok1
            v_max, ok2 = QInputDialog.getInt(self, 'Crop sample', f'{axis} max', value=v_max, min=v_min+1, max=v_max)
            return v_min, v_max, ok2

        z_min, z_max, ok = get_min_max(z_max, 'Z')
        if not ok:
            return
        y_min, y_max, ok = get_min_max(y_max, 'Y')
        if not ok:
            return
        x_min, x_max, ok = get_min_max(x_max, 'X')
        if not ok:
            return

        self.sample_crop_manager(dataset, sample, z_min, z_max, y_min, y_max, x_min, x_max)

    # ----------------------------------------
    # Wizards and forms manager
    # ----------------------------------------
    def sample_add_manager(self, choice_dataset, choice_sample, choice_image, choice_labels):
        self.blocking_task(target=self.manager.add_sample,
                           args=(choice_dataset, choice_sample, choice_image, choice_labels),
                           message='Loading sample(s)...',
                           target_end=self.dataset_update,
                           wait_end=False)

    def sample_crop_manager(self, choice_dataset, choice_sample, z_min, z_max, y_min, y_max, x_min, x_max):
        self.manager.crop_sample(choice_dataset, choice_sample, z_min, z_max, y_min, y_max, x_min, x_max)

    def dataset_new_manager(self, choice_dataset, choice_number_label):
        self.manager.new_dataset(choice_dataset, choice_number_label)

    # ----------------------------------------
    # Blocking task
    # ----------------------------------------
    def blocking_task(self, target, args, message, message_end=None, target_end=None, wait_end=True):
        if message_end is None:
            message_end = message + '\nDone !'
        # MessageBox
        self.bt_messagebox = QMessageBox(self)
        # self.bt_messagebox.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.bt_messagebox.setWindowTitle('DeepSCEM')
        self.bt_messagebox.setText(message + '\nClosing this message box will not terminate the current operation.')
        self.bt_messagebox.setVisible(True)
        # Disable UI (blocking task)
        self.setEnabled(False)
        self.bt_messagebox.setEnabled(False)
        # Thread and Worker
        self.bt_thread = QThread()
        self.bt_worker = GenericWorker(target=target, args=args)
        self.bt_worker.moveToThread(self.bt_thread)
        self.bt_thread.started.connect(self.bt_worker.run)
        self.bt_worker.finished.connect(self.bt_thread.quit)
        self.bt_worker.finished.connect(self.bt_worker.deleteLater)
        self.bt_thread.finished.connect(self.bt_thread.deleteLater)
        self.bt_thread.start()
        # Enable UI back when everything is done
        if target_end is not None:
            self.bt_thread.finished.connect(lambda: target_end())
        self.bt_thread.finished.connect(lambda: self.setEnabled(True))
        self.bt_thread.finished.connect(lambda: self.bt_messagebox.setText(message_end))
        self.bt_thread.finished.connect(lambda: self.bt_messagebox.setEnabled(True))
        if not wait_end:
            self.bt_thread.finished.connect(lambda: self.bt_messagebox.setVisible(False))


class GenericWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, target, args):
        super().__init__()
        self.target = target
        self.args = args

    def run(self):
        self.target(*self.args)
        self.finished.emit()
