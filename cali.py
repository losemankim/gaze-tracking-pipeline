import sys
import typing
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QMessageBox
import eyepass as main
import camera_calibration as camera
cali_class = uic.loadUiType("./cali.ui")[0]
class caliwidget(QWidget,cali_class):
    def __init__(self,monitor_info) :
        super().__init__() 
        self.setupUi(self) 
        self.monitor_info={'mm':None,'dpi':None}
        self.cam_calibration_No.clicked.connect(self.cam_calibration_No_clicked)
        self.cam_calibration_Yes.clicked.connect(self.cam_calibration_YES_clicked)
        
    def cam_calibration_No_clicked(self):
        self.close()
    def cam_calibration_YES_clicked(self):
        camera.record_video(width=1280, height=720, fps=30)
        cmd="ffmpeg -i output.mp4 -f image2 frames/output-%07d.png"
        camera.calibration('./frames', 30, debug=True)
