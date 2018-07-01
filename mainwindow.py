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
        MainWindow.resize(656, 439)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(60, 380, 141, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(260, 380, 141, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_3.setGeometry(QtCore.QRect(460, 380, 141, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_4.setGeometry(QtCore.QRect(260, 290, 141, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_5.setGeometry(QtCore.QRect(450, 290, 141, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_6.setGeometry(QtCore.QRect(60, 290, 141, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(70, 340, 241, 31))
        self.label.setObjectName("label")
        self.video_frame = QtWidgets.QLabel(self.centralWidget)
        self.video_frame.setGeometry(QtCore.QRect(90, 30, 471, 231))
        # self.label_2.setObjectName("label_2")
        self.video_frame.setObjectName("video_frame")
        MainWindow.setCentralWidget(self.centralWidget)
        self.pushButton.pressed.connect(lambda : inference("malaria"))
        self.pushButton_2.pressed.connect(lambda : inference("tuberculosis"))
        self.pushButton_3.pressed.connect(lambda : inference("intestinal"))
        self.pushButton_4.pressed.connect(lambda : capture())
        # self.pushButton_5.pressed.connect(lambda : )
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Malaria"))
        self.pushButton_2.setText(_translate("MainWindow", "Tuberculosis"))
        self.pushButton_3.setText(_translate("MainWindow", "Intestinal Parasites"))
        self.pushButton_4.setText(_translate("MainWindow", "Capture"))
        self.pushButton_5.setText(_translate("MainWindow", "Save Image"))
        self.pushButton_6.setText(_translate("MainWindow", "Load Image"))
        self.label.setText(_translate("MainWindow", "Check for :"))

class RecordVideo():
        def exit(self):
            self.camera.release()

        def nextFrameSlot(self):
            self.camera.set(3,1960)
            self.camera.set(4,1080)
            ret, frame = self.camera.read()
            
            img = get_qimage(frame) 
            pix = QtGui.QPixmap.fromImage(img)
            ui.video_frame.setPixmap(pix)
            inference() 

        def start_video(self):
            self.camera = cv2.VideoCapture(1)
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.nextFrameSlot)
            self.timer.start(1000./30)

        def stop_video(self):
            self.camera.release()
            self.timer.stop()



def get_qimage(image):
    height, width, colors = image.shape
    bytesPerLine = 3 * width
    QImage = QtGui.QImage
    image = QImage(image.tostring(),width,height,bytesPerLine,QImage.Format_RGB888)
    image = image.rgbSwapped()
    return image

def inference(name):
    input_shape,fileName=set_params(name)
    opts = set_opts(name)
    model=load_model_weights(input_shape,fileName)
    found = det.detect(imfile, model, opts)
    for f in found:
         f = f.astype(int)
         cv2.rectangle(im, (f[0],f[1]), (f[2],f[3]), (255,0,0), 6)


   
def capture():
    ui.textBrowser.clear()
    cap.camera=cv2.VideoCapture(1)
    frame = imutils.resize(frame, width=500)
    cv2.imwrite("Patient/ID"+str(),img_left)
    cap.camera.release()
    ui.textBrowser.clear()
    ui.textBrowser.append('Images are captured')


# if __name__ == "__main__":

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
cap=RecordVideo()
cap.flag=-1
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.setWindowTitle("Visible Iris Recognition")
MainWindow.show()
sys.exit(app.exec_())

