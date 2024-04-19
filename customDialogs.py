from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import window_newmodelui
import window_trainui
import window_predui
import window_evalui


class DialogNewModel(QtWidgets.QDialog, window_newmodelui.Ui_Dialog):
    def __init__(self, *args, **kwargs):
        super(DialogNewModel, self).__init__(*args, **kwargs)
        self.setupUi(self)

    def get(self):
        return (self.comboBox_dimension.currentText(),
                self.comboBox_architecture.currentText(),
                self.comboBox_backbone.currentText(),
                self.spinBox_kernel_size.value(),
                self.spinBox_block_filters.value(),
                self.spinBox_block_per_level.value(),
                self.comboBox_normalization.currentText(),
                self.spinBox_depth.value(),
                self.spinBox_outputs.value(),
                self.comboBox_output_activation.currentText())


class DialogTrain(QtWidgets.QDialog, window_trainui.Ui_Dialog):
    def __init__(self, datasets, models, *args, **kwargs):
        super(DialogTrain, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.comboBox_model.addItems(models)
        self.comboBox_dataset_train.addItems(datasets)
        self.comboBox_dataset_valid.addItems(datasets)

    def get(self):
        return (self.comboBox_model.currentIndex(),
                self.comboBox_dataset_train.currentText(),
                self.comboBox_dataset_valid.currentText(),
                self.comboBox_loss.currentText(),
                self.spinBox_batch_size.value(),
                self.spinBox_patch_size_z.value(),
                self.spinBox_patch_size_y.value(),
                self.spinBox_patch_size_x.value(),
                self.spinBox_steps_per_epoch.value(),
                self.spinBox_epochs.value(),
                self.spinBox_validation_steps.value(),
                self.checkBox_keep_best.isChecked(),
                self.checkBox_early_stopping.isChecked(),
                (self.checkBox_rot.isChecked(),  self.checkBox_flip.isChecked()))


class DialogPred(QtWidgets.QDialog, window_predui.Ui_Dialog):
    def __init__(self, datasets, models, *args, **kwargs):
        super(DialogPred, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.comboBox_model.addItems(models)
        self.comboBox_dataset.addItems(datasets)

    def get(self):
        return (self.comboBox_model.currentIndex(),
                self.comboBox_dataset.currentText(),
                self.spinBox_patch_size_z.value(),
                self.spinBox_patch_size_y.value(),
                self.spinBox_patch_size_x.value(),
                self.checkBox_overlapping.isChecked(),
                self.checkBox_threshold.isChecked(),
                self.doubleSpinBox_threshold.value())


class DialogEval(QtWidgets.QDialog, window_evalui.Ui_Dialog):
    def __init__(self, datasets, *args, **kwargs):
        super(DialogEval, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.comboBox_ref.addItems(datasets)
        self.comboBox_seg.addItems(datasets)

    def get(self):
        return (self.comboBox_ref.currentText(),
                self.comboBox_seg.currentText(),
                self.checkBox_F1.isChecked(),
                self.checkBox_IoU.isChecked())
