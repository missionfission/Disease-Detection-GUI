import cv2
import numpy as np
from model import load_model_weights, set_opts, set_params
from keras.preprocessing import image
import time
import pandas as pd
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
import sys
import time
from PyQt5 import QtCore, QtGui, QtWidgets
import sys


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(740, 600)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 520, 141, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(230, 520, 141, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_3.setGeometry(QtCore.QRect(410, 520, 141, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_4.setGeometry(QtCore.QRect(100, 320, 121, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_5.setGeometry(QtCore.QRect(290, 320, 121, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_6.setGeometry(QtCore.QRect(470, 320, 121, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_7.setGeometry(QtCore.QRect(290, 460, 84, 28))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(30, 480, 241, 31))
        self.label.setObjectName("label")
        self.video_frame = QtWidgets.QLabel(self.centralWidget)
        self.video_frame.setGeometry(QtCore.QRect(110, 20, 471, 231))
        self.video_frame.setObjectName("video_frame")
        self.lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit.setGeometry(QtCore.QRect(130, 420, 431, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralWidget)
        self.textBrowser.setGeometry(QtCore.QRect(130, 380, 431, 31))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralWidget)
        self.pushButton.pressed.connect(lambda: inference("malaria"))
        self.pushButton_2.pressed.connect(lambda: inference("tuberculosis"))
        self.pushButton_3.pressed.connect(lambda: inference("intestinal"))
        self.pushButton_4.pressed.connect(lambda: capture())
        self.pushButton_5.pressed.connect(lambda: save_image())
        self.pushButton_6.pressed.connect(lambda: load_image())
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 680, 25))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menuBar)
        self.actionCapture = QtWidgets.QAction(MainWindow)
        self.actionCapture.setObjectName("actionCapture")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionStop_Camera = QtWidgets.QAction(MainWindow)
        self.actionStop_Camera.setObjectName("actionStop_Camera")
        self.actionStart_Camera = QtWidgets.QAction(MainWindow)
        self.actionStart_Camera.setObjectName("actionStart_Camera")
        self.menuFile.addAction(self.actionCapture)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionClose)
        self.menuFile.addAction(self.actionStop_Camera)
        self.menuFile.addAction(self.actionStart_Camera)
        self.menuBar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Malaria"))
        self.pushButton_2.setText(_translate("MainWindow", "Tuberculosis"))
        self.pushButton_3.setText(_translate(
            "MainWindow", "Intestinal Parasites"))
        self.pushButton_4.setText(_translate(
            "MainWindow", "Start VideoCapture"))
        self.pushButton_5.setText(_translate("MainWindow", "Save Image"))
        self.pushButton_6.setText(_translate("MainWindow", "Load Image"))
        self.label.setText(_translate("MainWindow", "Check for :"))
        self.pushButton_7.setText(_translate("MainWindow", "OK"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionCapture.setText(_translate("MainWindow", "Capture"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionStop_Camera.setText(_translate("MainWindow", "Stop Camera"))
        self.actionStart_Camera.setText(
            _translate("MainWindow", "Start Camera"))


#######################################################################################################################################


class RecordVideo():
    def exit(self):
        self.camera.release()

    def nextFrameSlot(self):
        self.camera.set(3, 1960)
        self.camera.set(4, 1080)
        ret, frame = self.camera.read()
        img = get_qimage(frame)
        pix = QtGui.QPixmap.fromImage(img)
        ui.video_frame.setPixmap(pix)

    def save(self):
        self.frame()

    def start_video(self):
        self.camera = cv2.VideoCapture(1)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000. / 30)

    def stop_video(self):
        self.camera.release()
        self.timer.stop()

#######################################################################################################################################


def save_image():
    cap.stop_video()
    ui.textBrowser.clear()
    ui.textBrowser.append('Enter the fileName')
    ui.textBrowser.clear()
    while(!ui.pushButton_7.clicked):
        continue
    filename = ui.lineEdit.text()
    cv2.imwrite(filename, cap.frame)

#######################################################################################################################################


def load_image():
    cap.stop_video()
    ui.textBrowser.clear()
    ui.textBrowser.append('Enter the fileName')
    while(!ui.pushButton_7.clicked):
        continue
    filename = ui.lineEdit.text()
    ui.textBrowser.clear()
    frame = cv2.imread(filename)
    img = get_qimage(frame)
    pix = QtGui.QPixmap.fromImage(img)
    ui.video_frame.setPixmap(pix)

#######################################################################################################################################


def get_qimage(image):
    height, width, colors = image.shape
    bytesPerLine = 3 * width
    QImage = QtGui.QImage
    image = QImage(image.tostring(), width, height,
                   bytesPerLine, QImage.Format_RGB888)
    image = image.rgbSwapped()
    return image

#######################################################################################################################################


def inference(name):
    input_shape, fileName = set_params(name)
    opts = set_opts(name)
    model = load_model_weights(input_shape, fileName)
    # TODO add some dummy files for now to show the output
    #     imfile="   "
    found = det.detect(imfile, model, opts)
    for f in found:
        f = f.astype(int)
        cv2.rectangle(im, (f[0], f[1]), (f[2], f[3]), (255, 0, 0), 6)

#######################################################################################################################################


def capture():
    ui.textBrowser.clear()
    cap.start_video()


#######################################################################################################################################
# if __name__ == "__main__":
import pdb
import sys
# pdb.set_trace()
app = QtWidgets.QApplication(sys.argv)
print("app initiated")
MainWindow = QtWidgets.QMainWindow()
cap = RecordVideo()
# cap.flag=-1
# print("D")
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.setWindowTitle("Pathogen Detection")
MainWindow.show()
sys.exit(app.exec_())
