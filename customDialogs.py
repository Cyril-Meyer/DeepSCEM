from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import window_newmodelui
import window_trainui


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
        self.comboBox_dataset_test.addItems(datasets)

    def get(self):
        return (self.comboBox_model.currentIndex(),
                self.comboBox_dataset_train.currentText(),
                self.comboBox_dataset_valid.currentText(),
                self.comboBox_dataset_test.currentText(),
                self.comboBox_loss.currentText(),
                self.spinBox_batch_size.value(),
                self.spinBox_patch_size_z.value(),
                self.spinBox_patch_size_y.value(),
                self.spinBox_patch_size_x.value(),
                self.spinBox_steps_per_epoch.value(),
                self.spinBox_epochs.value(),
                self.spinBox_validation_steps.value(),
                self.checkBox_keep_best.isChecked(),
                self.checkBox_early_stopping.isChecked())
