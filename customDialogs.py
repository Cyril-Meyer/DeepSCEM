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
    def __init__(self, *args, **kwargs):
        super(DialogTrain, self).__init__(*args, **kwargs)
        self.setupUi(self)
