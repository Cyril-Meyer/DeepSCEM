from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from window_newmodelui import Ui_Dialog


class DialogNewModel(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, *args, obj=None, **kwargs):
        super(DialogNewModel, self).__init__(*args, **kwargs)
        self.setupUi(self)
