from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class InteractiveQLabel(QLabel):
    mousePress = pyqtSignal(QMouseEvent)
    mouseRelease = pyqtSignal(QMouseEvent)
    mouseMove = pyqtSignal(QMouseEvent)
    wheel = pyqtSignal(QWheelEvent)

    def mousePressEvent(self, event):
        self.mousePress.emit(event)
        QLabel.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.mouseRelease.emit(event)
        QLabel.mouseReleaseEvent(self, event)

    def mouseMoveEvent(self, event):
        self.mouseMove.emit(event)
        QLabel.mouseMoveEvent(self, event)

    def wheelEvent(self, event):
        self.wheel.emit(event)
        QLabel.wheelEvent(self, event)


class QTreeWidgetWithDrop(QTreeWidget):
    ok_ext = ['hdf5']
    drop = QtCore.pyqtSignal(QUrl)

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            for url in event.mimeData().urls():
                if QFileInfo(url.toLocalFile()).suffix().lower() in self.ok_ext:
                    self.drop.emit(url)
        else:
            event.ignore()

    def mimeTypes(self):
        return ['text/uri-list']
