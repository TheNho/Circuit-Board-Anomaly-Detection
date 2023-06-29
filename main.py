# -- coding: utf-8 --
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from CamOperation_class import CameraOperation
from MvImport.MvCameraControl_class import *
from MvImport.MvErrorDefine_const import *
from MvImport.CameraParams_header import *
from PyUICBasicDemo import Ui_MainWindow

import cv2
import threading
import torch
import numpy as np
import os
from Padim.padim import Padim
from Padim.visualization import heatmap_image, boundary_image
from PIL import Image
from Padim.utils import standard_image_transform, classification
import snap7

class Images_Handle:
    def __init__(self, img):
        self.image = img
    def get(self):
        return self.image
    def set(self, img):
        self.image = img

def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()

def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr

if __name__ == "__main__":

    global deviceList
    deviceList = MV_CC_DEVICE_INFO_LIST()
    global cam
    cam = MvCamera()
    global nSelCamIndex
    nSelCamIndex = 0
    global obj_cam_operation
    obj_cam_operation = 0
    global isOpen
    isOpen = False
    global isGrabbing
    isGrabbing = False
    global isCalibMode  
    isCalibMode = True
    img_handle = Images_Handle(None)

    def run():
        import struct
        import time
        address = 100  # starting address in PLC
        length = 4  # double word

        def check_equal_value(a,b, epsilon=0.2):
            if b-epsilon <= a <= b+epsilon:
                return True
            return False

        def readMemory(start_address,length):
            try:
                reading = plc_s7.read_area(snap7.types.Areas.MK, 0, start_address, length)
                value = struct.unpack('>f', reading)  # big-endian
                return value[0]
            except:
                QMessageBox.warning(mainWindow, "Error", "Cannot read plc memory", QMessageBox.Ok)
                return -1
            
        def writeMemory(start_address,length,value):
            try:
                plc_s7.mb_write(start_address, length, bytearray(struct.pack('>f', value)))  # big-endian
            except:
                QMessageBox.warning(mainWindow, "Error", "Cannot write value to plc", QMessageBox.Ok)
        
        def commuicate_with_plc():
            while isGrabbing and is_padim_model_inited and plc_s7.get_connected():
                ui.result_text.setText('')
                get_value = readMemory(address, length)
                if check_equal_value(get_value, 1):
                    t = time.time()
                    try: # get threshold value
                        THRESH = float(ui.Threshold.text())
                    except:
                        print('Threshold value error!')
                        time.sleep(0.5)
                        continue
                    
                    img = img_handle.get().copy() #HxWx3 type RGB
                    
                    image = Image.fromarray(img)

                    transformed_images = standard_image_transform(image).unsqueeze(0) #1xCxHxW
                    # print('transformed image shape', transformed_images.shape)

                    # get model output
                    image_scores, score_maps = padim_model.predict(transformed_images)
                    print("Model predict in", round(time.time()-t, 4), "s")
                    # image_scores: tensor 1x1
                    # score_maps: tensor 1xHxW
                    
                    # draw output
                    if ui.radio_heatmap.isChecked():
                        im_show = heatmap_image(img, score_maps[0], min_v=0, max_v=THRESH, alpha=0.5) # HxWxC
                    else:
                        score_map_classifications = classification(score_maps[0], THRESH)
                        im_show = boundary_image(img, score_map_classifications)
                    im_show = cv2.resize(im_show, (500,400))
                    qImg = QtGui.QImage(im_show, 
                                        im_show.shape[1], 
                                        im_show.shape[0], 
                                        im_show.shape[1] * 3, 
                                        QtGui.QImage.Format_RGB888)
                    pix = QtGui.QPixmap(qImg)
                    ui.img_result.setPixmap(pix)
                    
                    # send result to plc
                    send_value = 2.0
                    text = 'GOOD'
                    if image_scores.item() > THRESH:
                        text = 'NOT GOOD'
                        send_value = 3.0
                    ui.result_text.setText(text)
                    writeMemory(address, length, send_value)
                time.sleep(0.1)
        
        if not isGrabbing:
            QMessageBox.warning(mainWindow, "Error", "Start Grabbing first!", QMessageBox.Ok)
            return
        if not plc_s7.get_connected():
            QMessageBox.warning(mainWindow, "Error", "Plc is not connected", QMessageBox.Ok)
            return
        if not is_padim_model_inited:
            QMessageBox.warning(mainWindow, "Error", "Padim model is not initialized", QMessageBox.Ok)
            return
        # run main function
        plc_thread_handle = threading.Thread(target=commuicate_with_plc)
        plc_thread_handle.start()

    global padim_model
    global is_padim_model_inited
    is_padim_model_inited = False
    MODEL_DATA_PATH = './Padim/saved_model'
    def init_padim_model():
        global padim_model
        global is_padim_model_inited
        # Check threshold value
        try:
            THRESH = float(ui.Threshold.text())
        except:
            QMessageBox.warning(mainWindow, "Error", "error THRESH value", QMessageBox.Ok)
            return
        # check load model
        try:
            mean = torch.load(os.path.join(MODEL_DATA_PATH, 'mean.pt'))
            cov_inv = torch.load(os.path.join(MODEL_DATA_PATH, 'cov_inv.pt'))
            channel_indices = torch.load(os.path.join(MODEL_DATA_PATH, 'channel_indices.pt'))
            padim_model = Padim(backbone='resnet18', 
                                mean=mean, 
                                cov_inv=cov_inv, 
                                channel_indices = channel_indices,
                                device=torch.device('cpu'))
            is_padim_model_inited = True
            print('Load model success!')
        except:
            QMessageBox.warning(mainWindow, "Error", "Cannot initialize model", QMessageBox.Ok)
            is_padim_model_inited = False
        
    global plc_s7
    plc_s7 = snap7.client.Client()
    def init_plc():
        if not plc_s7.get_connected():
            try:
                plc_s7.connect(ui.ip_address.text(), 0, 1)
                print('Connect to plc success!')
            except:
                QMessageBox.warning(mainWindow, "Error", "Cannot connect to plc", QMessageBox.Ok)
        else:
            print("Plc is already connected!")
        
    def xFunc(event):
        global nSelCamIndex
        nSelCamIndex = TxtWrapBy("[", "]", ui.ComboDevices.get())
    # en:enum devices
    def enum_devices():
        global deviceList
        global obj_cam_operation
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0:
            strError = "Enum devices fail! ret = :" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            return ret
        if deviceList.nDeviceNum == 0:
            QMessageBox.warning(mainWindow, "Info", "Find no device", QMessageBox.Ok)
            return ret
        print("Find %d devices!" % deviceList.nDeviceNum)
        devList = []
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                    if 0 == per:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print("device user define name: %s" % chUserDefinedName)
                chModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if 0 == per:
                        break
                    chModelName = chModelName + chr(per)
                print("device model name: %s" % chModelName)
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                devList.append(
                    "[" + str(i) + "]GigE: " + chUserDefinedName + " " + chModelName + "(" + str(nip1) + "." + str(
                        nip2) + "." + str(nip3) + "." + str(nip4) + ")")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                chUserDefinedName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                    if per == 0:
                        break
                    chUserDefinedName = chUserDefinedName + chr(per)
                print("device user define name: %s" % chUserDefinedName)

                chModelName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if 0 == per:
                        break
                    chModelName = chModelName + chr(per)
                print("device model name: %s" % chModelName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)
                devList.append("[" + str(i) + "]USB: " + chUserDefinedName + " " + chModelName
                               + "(" + str(strSerialNumber) + ")")
        ui.ComboDevices.clear()
        ui.ComboDevices.addItems(devList)
        ui.ComboDevices.setCurrentIndex(0)

    # en:open device
    def open_device():
        global deviceList
        global nSelCamIndex
        global obj_cam_operation
        global isOpen
        if isOpen:
            QMessageBox.warning(mainWindow, "Error", 'Camera is Running!', QMessageBox.Ok)
            return MV_E_CALLORDER
        nSelCamIndex = ui.ComboDevices.currentIndex()
        if nSelCamIndex < 0:
            QMessageBox.warning(mainWindow, "Error", 'Please select a camera!', QMessageBox.Ok)
            return MV_E_CALLORDER
        obj_cam_operation = CameraOperation(cam, deviceList, nSelCamIndex)
        ret = obj_cam_operation.Open_device()
        if 0 != ret:
            strError = "Open device failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            isOpen = False
        else:
            set_continue_mode()
            get_param()
            isOpen = True
            enable_controls()

    # en:Start grab image
    def start_grabbing():
        global obj_cam_operation
        global isGrabbing
        ret = obj_cam_operation.Start_grabbing(ui.widgetDisplay.winId(), img_handle)
        if ret != 0:
            strError = "Start grabbing failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            isGrabbing = False
        else:
            isGrabbing = True
            enable_controls()

    # en:Stop grab image
    def stop_grabbing():
        global obj_cam_operation
        global isGrabbing
        ret = obj_cam_operation.Stop_grabbing()
        if ret != 0:
            strError = "Stop grabbing failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            isGrabbing = False
            enable_controls()

    # Close device
    def close_device():
        global isOpen
        global isGrabbing
        global obj_cam_operation
        if isOpen:
            obj_cam_operation.Close_device()
            isOpen = False
        isGrabbing = False
        enable_controls()
    
    # en:set trigger mode
    def set_continue_mode():
        strError = None
        ret = obj_cam_operation.Set_trigger_mode(False)
        if ret != 0:
            strError = "Set continue mode failed ret:" + ToHexStr(ret) + " mode is error" #+ str(is_trigger_mode)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
    # en:set software trigger mode
    def set_software_trigger_mode():

        ret = obj_cam_operation.Set_trigger_mode(True)
        if ret != 0:
            strError = "Set trigger mode failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            ui.radioContinueMode.setChecked(False)
            ui.radioTriggerMode.setChecked(True)
            ui.bnSoftwareTrigger.setEnabled(isGrabbing)
    # en:set trigger software
    def trigger_once():
        ret = obj_cam_operation.Trigger_once()
        if ret != 0:
            strError = "TriggerSoftware failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

    # en:save image
    def save_bmp():
        ret = obj_cam_operation.Save_Bmp()
        if ret != MV_OK:
            strError = "Save BMP failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            print("Save BMP image success!")
    def save_jpg():
        ret = obj_cam_operation.Save_jpg()
        if ret != MV_OK:
            strError = "Save JPG failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            print("Save JPG image success!")
    
    # en:get param
    def get_param():
        ret = obj_cam_operation.Get_parameter()
        if ret != MV_OK:
            strError = "Get param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            ui.edtExposureTime.setText("{0:.2f}".format(obj_cam_operation.exposure_time))
            ui.edtGain.setText("{0:.2f}".format(obj_cam_operation.gain))
            ui.edtFrameRate.setText("{0:.2f}".format(obj_cam_operation.frame_rate))

    # en:set param
    def set_param():
        frame_rate = ui.edtFrameRate.text()
        exposure = ui.edtExposureTime.text()
        gain = ui.edtGain.text()
        ret = obj_cam_operation.Set_parameter(frame_rate, exposure, gain)
        if ret != MV_OK:
            strError = "Set param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

        return MV_OK

    # en:set enable status
    def enable_controls():
        global isGrabbing
        global isOpen

        # group
        ui.groupGrab.setEnabled(isOpen)
        ui.groupParam.setEnabled(isOpen)

        ui.bnOpen.setEnabled(not isOpen)
        ui.bnClose.setEnabled(isOpen)

        ui.bnStart.setEnabled(isOpen and (not isGrabbing))
        ui.bnStop.setEnabled(isOpen and isGrabbing)

        ui.bnSaveImageBMP.setEnabled(isOpen and isGrabbing)
        ui.bnSaveImageJPG.setEnabled(isOpen and isGrabbing)

    # en: Init app, bind ui and api
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    ui.bnEnum.clicked.connect(enum_devices)
    ui.bnOpen.clicked.connect(open_device)
    ui.bnClose.clicked.connect(close_device)
    ui.bnStart.clicked.connect(start_grabbing)
    ui.bnStop.clicked.connect(stop_grabbing)
    ui.bnGetParam.clicked.connect(get_param)
    ui.bnSetParam.clicked.connect(set_param)
    ui.bnSaveImageBMP.clicked.connect(save_bmp)
    ui.bnSaveImageJPG.clicked.connect(save_jpg)
    ui.btn_load_padim.clicked.connect(init_padim_model)
    ui.btn_connect_plc.clicked.connect(init_plc)
    ui.btn_run.clicked.connect(run)
    mainWindow.show()

    app.exec_()

    close_device()

    sys.exit()
