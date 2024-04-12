# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/logo/icons/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.mainLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.mainLayout.setObjectName("mainLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_dataset = QtWidgets.QHBoxLayout()
        self.horizontalLayout_dataset.setObjectName("horizontalLayout_dataset")
        self.label_dataset = QtWidgets.QLabel(self.centralwidget)
        self.label_dataset.setObjectName("label_dataset")
        self.horizontalLayout_dataset.addWidget(self.label_dataset)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_dataset.addItem(spacerItem)
        self.pushButton_dataset_load = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_dataset_load.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/icons/outline_playlist_add_black_48dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_dataset_load.setIcon(icon1)
        self.pushButton_dataset_load.setObjectName("pushButton_dataset_load")
        self.horizontalLayout_dataset.addWidget(self.pushButton_dataset_load)
        self.pushButton_dataset_unload = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_dataset_unload.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/icons/outline_playlist_remove_black_48dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_dataset_unload.setIcon(icon2)
        self.pushButton_dataset_unload.setObjectName("pushButton_dataset_unload")
        self.horizontalLayout_dataset.addWidget(self.pushButton_dataset_unload)
        self.pushButton_sample_add = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_sample_add.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/icons/outline_add_photo_alternate_black_48dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_sample_add.setIcon(icon3)
        self.pushButton_sample_add.setObjectName("pushButton_sample_add")
        self.horizontalLayout_dataset.addWidget(self.pushButton_sample_add)
        self.pushButton_sample_remove = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_sample_remove.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/icons/outline_delete_black_48dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_sample_remove.setIcon(icon4)
        self.pushButton_sample_remove.setObjectName("pushButton_sample_remove")
        self.horizontalLayout_dataset.addWidget(self.pushButton_sample_remove)
        self.pushButton_dataset_saveas = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_dataset_saveas.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/icons/outline_save_black_48dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_dataset_saveas.setIcon(icon5)
        self.pushButton_dataset_saveas.setObjectName("pushButton_dataset_saveas")
        self.horizontalLayout_dataset.addWidget(self.pushButton_dataset_saveas)
        self.verticalLayout.addLayout(self.horizontalLayout_dataset)
        self.treeWidget_dataset = QtWidgets.QTreeWidget(self.centralwidget)
        self.treeWidget_dataset.setObjectName("treeWidget_dataset")
        self.verticalLayout.addWidget(self.treeWidget_dataset)
        self.horizontalLayout_model = QtWidgets.QHBoxLayout()
        self.horizontalLayout_model.setObjectName("horizontalLayout_model")
        self.label_model = QtWidgets.QLabel(self.centralwidget)
        self.label_model.setObjectName("label_model")
        self.horizontalLayout_model.addWidget(self.label_model)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_model.addItem(spacerItem1)
        self.pushButton_model_add = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_model_add.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/icons/outline_add_black_48dp.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_model_add.setIcon(icon6)
        self.pushButton_model_add.setObjectName("pushButton_model_add")
        self.horizontalLayout_model.addWidget(self.pushButton_model_add)
        self.verticalLayout.addLayout(self.horizontalLayout_model)
        self.listWidget_model = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_model.setObjectName("listWidget_model")
        self.verticalLayout.addWidget(self.listWidget_model)
        self.mainLayout.addLayout(self.verticalLayout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_mainview = InteractiveQLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_mainview.sizePolicy().hasHeightForWidth())
        self.label_mainview.setSizePolicy(sizePolicy)
        self.label_mainview.setAlignment(QtCore.Qt.AlignCenter)
        self.label_mainview.setObjectName("label_mainview")
        self.gridLayout.addWidget(self.label_mainview, 0, 0, 1, 1)
        self.horizontalSlider_z = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_z.setMaximum(0)
        self.horizontalSlider_z.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_z.setObjectName("horizontalSlider_z")
        self.gridLayout.addWidget(self.horizontalSlider_z, 1, 0, 1, 1)
        self.mainLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_dataset = QtWidgets.QAction(MainWindow)
        self.actionLoad_dataset.setObjectName("actionLoad_dataset")
        self.actionNew_dataset = QtWidgets.QAction(MainWindow)
        self.actionNew_dataset.setObjectName("actionNew_dataset")
        self.actionSave_dataset = QtWidgets.QAction(MainWindow)
        self.actionSave_dataset.setObjectName("actionSave_dataset")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.menuFile.addAction(self.actionNew_dataset)
        self.menuFile.addAction(self.actionLoad_dataset)
        self.menuFile.addAction(self.actionSave_dataset)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButton_dataset_load.clicked.connect(MainWindow.dataset_load_clicked) # type: ignore
        self.actionLoad_dataset.triggered.connect(MainWindow.dataset_load_clicked) # type: ignore
        self.pushButton_sample_add.clicked.connect(MainWindow.sample_add_clicked) # type: ignore
        self.pushButton_dataset_unload.clicked.connect(MainWindow.dataset_unload_clicked) # type: ignore
        self.pushButton_sample_remove.clicked.connect(MainWindow.sample_remove_clicked) # type: ignore
        self.pushButton_dataset_saveas.clicked.connect(MainWindow.dataset_saveas_clicked) # type: ignore
        self.actionNew_dataset.triggered.connect(MainWindow.dataset_new_clicked) # type: ignore
        self.actionSave_dataset.triggered.connect(MainWindow.dataset_saveas_clicked) # type: ignore
        self.label_mainview.mouseMove['QMouseEvent'].connect(MainWindow.mainview_mouse_event) # type: ignore
        self.label_mainview.mousePress['QMouseEvent'].connect(MainWindow.mainview_mouse_event) # type: ignore
        self.label_mainview.mouseRelease['QMouseEvent'].connect(MainWindow.mainview_mouse_event) # type: ignore
        self.treeWidget_dataset.itemSelectionChanged.connect(MainWindow.dataset_item_selection_changed) # type: ignore
        self.actionAbout.triggered.connect(MainWindow.menu_help_about_triggered) # type: ignore
        self.treeWidget_dataset.itemChanged['QTreeWidgetItem*','int'].connect(MainWindow.dataset_item_selection_changed) # type: ignore
        self.horizontalSlider_z.valueChanged['int'].connect(MainWindow.mainview_slider_changed) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DeepSCEM"))
        self.label_dataset.setText(_translate("MainWindow", "Dataset"))
        self.pushButton_dataset_load.setToolTip(_translate("MainWindow", "Load dataset"))
        self.pushButton_dataset_load.setStatusTip(_translate("MainWindow", "Load dataset"))
        self.pushButton_dataset_unload.setToolTip(_translate("MainWindow", "Unload dataset"))
        self.pushButton_dataset_unload.setStatusTip(_translate("MainWindow", "Unload dataset"))
        self.pushButton_sample_add.setToolTip(_translate("MainWindow", "Add sample"))
        self.pushButton_sample_add.setStatusTip(_translate("MainWindow", "Add sample"))
        self.pushButton_sample_remove.setToolTip(_translate("MainWindow", "Remove sample"))
        self.pushButton_sample_remove.setStatusTip(_translate("MainWindow", "Remove sample"))
        self.pushButton_dataset_saveas.setToolTip(_translate("MainWindow", "Save dataset as..."))
        self.pushButton_dataset_saveas.setStatusTip(_translate("MainWindow", "Save dataset as..."))
        self.treeWidget_dataset.headerItem().setText(0, _translate("MainWindow", "Dataset"))
        self.label_model.setText(_translate("MainWindow", "Models"))
        self.label_mainview.setText(_translate("MainWindow", "MainView"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionLoad_dataset.setText(_translate("MainWindow", "Load dataset"))
        self.actionLoad_dataset.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionNew_dataset.setText(_translate("MainWindow", "New dataset"))
        self.actionNew_dataset.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionSave_dataset.setText(_translate("MainWindow", "Save dataset as"))
        self.actionSave_dataset.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionAbout.setShortcut(_translate("MainWindow", "Ctrl+Space"))
from customQtWidgets import InteractiveQLabel
import mainwindow_rc
