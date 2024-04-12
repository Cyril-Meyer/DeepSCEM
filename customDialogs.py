from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from window_newmodelui import Ui_Dialog


class DialogNewModel(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, *args, obj=None, **kwargs):
        super(DialogNewModel, self).__init__(*args, **kwargs)
        self.setupUi(self)

    def get(self):
        return (self.comboBox_dimension.currentText(),
                self.comboBox_architecture.currentText(),
                self.spinBox_block.value(),
                self.spinBox_depth.value(),
                self.spinBox_outputs.value(),
                self.comboBox_output_activation.currentText())
