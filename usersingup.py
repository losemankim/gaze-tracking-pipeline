import sys
import typing
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QMessageBox
import eyepass as main
class signupwidget(QWidget,main.signup_class):
    def __init__(self,callback):
        super().__init__()
        self.setupUi(self)
        self.usersignup_YES.clicked.connect(self.usersignup_YES_clicked)
        self.usersignup_NO.clicked.connect(self.usersignup_NO_clicked)
        self.callback = callback
        self.userinfo = {'id': None, 'email': None, 'phone': None}
        #자기혼자 꺼지는거 방지
        self.setWindowModality(QtCore.Qt.ApplicationModal)
    def usersignup_YES_clicked(self):
        self.userinfo['id']=self.user_name_InputTextbox.toPlainText()
        self.userinfo['email']=self.user_email_Input_Textbox.toPlainText()
        self.userinfo['phone']=self.user_phone_inputTextbox.toPlainText()
        self.callback(self.userinfo)
        self.close()
    def usersignup_NO_clicked(self):
        self.close()