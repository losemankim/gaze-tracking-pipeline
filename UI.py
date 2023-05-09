import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QMessageBox
import threading
import camera_calibration as camera
#ffmpeg를 이용해 파일을 자르는 모듈
import subprocess
import play as main
form_class = uic.loadUiType("./dialog.ui")[0]

form_class2 = uic.loadUiType("./dialog2.ui")[0]

class camwidgets(QWidget, form_class2):
    def __init__(self) -> None:

        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("카메라 설정")
        self.on.clicked.connect(self.on_click)
        self.lineEdit.setPlaceholderText("화면의 크기를 입력해주세요.(ex: 1920,1080)")

    def on_click(self):
        pixel=self.lineEdit.text()
        pixel2 = pixel.split(",")
        # camera.record_video(width=pixel2[0], height=pixel2[1], fps=30)ㄴ
        print("시작")
        camera.record_video(width=1280, height=720, fps=30)
        cmd="ffmpeg -i output.mp4 -f image2 frames/output-%07d.png"
        print("자르기")
        result=subprocess.run(cmd)
        print(result)
        print("이제 검증")
        camera.calibration('./frames', 30, debug=True)
        pass
        
        
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        #선언
        super().__init__()
        self.setupUi(self)
        #설정
        #창크기 변경 금지
        self.setFixedSize(800, 600)
        
        #스레드 생성
        self.thread = threading.Thread()
        #연결
        self.start_btn.clicked.connect(self.start)
        self.cam_btn.clicked.connect(self.cam)
        self.setpw_btn.clicked.connect(self.setpw)
        self.exit_btn.clicked.connect(self.exit)
        self.help_btn.clicked.connect(self.help)
        self.sel_btn.clicked.connect(self.sel)
        self.del_btn.clicked.connect(self.dele)
        self.add_btn.clicked.connect(self.add)
        #placeholder의 텍스트를 설정해준다.
        self.moni.setPlaceholderText("모니터의 크기를 입력해주세요.(ex: 1920,1080)")
        
    #15인치 = 381mm 16:9 비율이라면 243mm, 137mm 일때 15인치 모니터의 크기가
    def start(self) :
        print("start")
        password=[1,2,3,4]
        text=main.pw_make(calibration_matrix_path="./calibration_matrix.yaml",model_path="../p03.ckpt",monitor_mm=(243,137),monitor_pixels=(1920,1080),visualize_laser_pointer=True,password=password)
        #메세지박스를 띄운다.
        print(text)
        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Warning)
        self.msgBox.setText(text)#이것은 메시지 상자의 제목 텍스트를 설정합니다.
        self.msgBox.setWindowTitle("인증")#이것은 메시지 상자의 제목 텍스트를 설정합니다.
        self.msgBox.setStandardButtons(QMessageBox.Ok)#이것은 메시지 상자의 표준 버튼을 설정합니다.
        self.msgBox.exec_()
        if(self.msgBox.Ok):
            print("ok")
        
        pass
    def cam(self):
        print("cam")
        #카메라 설정창을 띄운다.
        self.camwidget = camwidgets()
        self.camwidget.show()
        pass
    def setpw(self) :
        
        
        pass
    def exit(self) :#완료
        #프로그램을 종료한다.
        self.close()
        pass
    def help(self) :
        pass
    def add(self) :
        pass
    def dele(self) :
        pass
    def sel(self) :
        pass
    
    
        
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()