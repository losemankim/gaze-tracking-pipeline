import sys
import os
import typing
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QMessageBox
import camera_calibration as camera
#ffmpeg를 이용해 파일을 자르는 모듈
import subprocess
import play as main
import usersingup as signup
import cali as cali


main_class = uic.loadUiType("./main.ui")[0]
cali_class = uic.loadUiType("./cali.ui")[0]
signup_class = uic.loadUiType("./usersignup.ui")[0]


"""
setupUi(self)를 이용해 UI를 설정해준다. 이 함수는 UI파일을 로드하고, UI파일에 있는 위젯들을 인스턴스화 해준다.
"""
class mainwidget(QWidget,main_class):
    def __init__(self) :
        super().__init__()
        #./user 폴더에 있는 모든 파일을 가져온다.
        self.saved_user_list=os.listdir('./user')
        self.setupUi(self)
        self.monitor_info={'mm':None,'dpi':None}
        self.User_reg_btn.clicked.connect(self.user_reg)
        self.Moniter_mm_label
        self.Moniter_mm_InputTextbox.setPlaceholderText("모니터의 크기를 입력해주세요.(ex: 381, 243)")
        self.Moniter_mm_InputTextbox.textChanged.connect(self.handle_monitor_info)        
        self.Moniter_dpi_label
        self.Moniter_dpi_InputTextbox.setPlaceholderText("모니터의 해상도를 입력해주세요.(ex: 1920, 1080)")
        self.Moniter_dpi_InputTextbox.textChanged.connect(self.handle_monitor_info)
        self.Start_cert_btn.clicked.connect(self.start_cert)
        self.Userlist_box.itemClicked.connect(self.handle_select_user)
        self.User_del_btn.clicked.connect(self.Userlist_box_del)
        self.Cam_Calibration_btn.clicked.connect(self.Cam_Calibration_btn_clicked)
        self.Program_exit_btn.clicked.connect(self.close)
        self.Help_btn
        self.userinfo=None
        self.username=None
        self.signup_window = signup.signupwidget(self.handle_userinfo)
        self.cali_window = cali.caliwidget(self.monitor_info)
        for i in range(len(self.saved_user_list)):
            self.saved_user_list[i]=self.saved_user_list[i].replace('.txt','')
        self.Userlist_box.addItems(self.saved_user_list)
        
    def handle_select_user(self):
        self.username=self.Userlist_box.currentItem()
    def start_cert(self):
        if(self.username==None):
            print("사용자를 선택해주세요.")
            #경고창 띄우기
            QMessageBox.about(self, "경고", "사용자를 선택해주세요.")
        else:
            print(self.username.text())
            #username.txt파일을 찾는다
            try :
                self.username=self.username.text().replace('\n','')
                f=open('./user/'+self.username+'.txt','r')
            except FileNotFoundError:
                print("파일이 존재하지 않습니다.")
                QMessageBox.about(self, "경고", "파일이 존재하지 않습니다\n 비밀번호를 설정합니다.")
                # text=main.pw_make(calibration_matrix_path="./calibration_matrix.yaml",model_path="../p03.ckpt",monitor_mm=(243,137),monitor_pixels=(1920,1080),visualize_laser_pointer=True,password=None)
                f=open('./user/'+self.username+'.txt','w')
                f.write('test')
                # f.write(self.username.text()+'\n'+text+'\n'+str(self.user_info['email']),'\n'+str(self.user_info['phone']))
                
    def user_reg(self):
        self.signup_window.show()
    def Cam_Calibration_btn_clicked(self):
        if(self.monitor_info['mm']==None or self.monitor_info['dpi']==None):
            print("모니터 정보를 입력해주세요.")
            #경고창 띄우기
            QMessageBox.about(self, "경고", "모니터 정보를 입력해주세요.")
        else:
            self.cali_window.show()
            
    def Userlist_box_del(self):#선택된 아이템을 삭제한다.
        self.Userlist_box.takeItem(self.Userlist_box.currentRow())
    def handle_userinfo(self,userinfo): #handle_userinfo이 함수를 이용하여 회원가입 정보를 처리한다.
        self.userinfo = userinfo
        self.Userlist_box.addItem(userinfo['id'])
    def handle_monitor_info(self):
        self.monitor_info["mm"]=self.Moniter_mm_InputTextbox.toPlainText()
        self.monitor_info["dpi"]=self.Moniter_dpi_InputTextbox.toPlainText()
        print(self.monitor_info)
    def close(self):
        sys.exit()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = mainwidget()
    myWindow.show()
    #앱실행
    app.exec_() 
        
    