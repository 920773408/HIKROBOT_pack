import math
import multiprocessing
import queue
import threading
import time
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from PIL import Image, ImageDraw, ImageFont

from MvCameraControl_class import *


class CameraOperation(object):

    def __init__(self, obj_cam, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None, b_thread_opened=False, st_frame_info=None, b_save_bmp=False, b_save_jpg=False, buf_save_image=None):

        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_opened = b_thread_opened
        self.st_frame_info = MV_FRAME_OUT_INFO_EX()
        self.save_images_flag = threading.Event()  # 用于标记是否需要保存图像
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.buf_save_image_len = 0
        self.h_thread_handle = h_thread_handle
        self.buf_lock = threading.Lock()  # 取图和存图的buffer锁
        self.exit_flag = 0
        self.frame_count = 0
        self.lost_frame_count = 0

    # 转为16进制字符串
    def to_hex_str(self, num):
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

    # 打开相机
    def open_device(self):
        if self.b_open_device is False:
            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.b_open_device = False
                self.b_thread_opened = False
                return ret
            self.b_open_device = True
            self.b_thread_opened = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: packet size is invalid[%d]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("warning: get acquisition frame rate enable fail! ret[0x%x]" % ret)
            return 0
        return 0

    def configure_master_camera(self):
        # 设置主相机参数
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print('Disable trigger failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetEnumValue("LineSelector", MV_TRIGGER_SOURCE_LINE1)
        if ret != 0:
            print('Set Line Selector failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetBoolValue("StrobeEnable", True)
        if ret != 0:
            print('Enable Strobe failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 15.0)
        if ret != 0:
            print('Set Frame Rate failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", 50000)
        if ret != 0:
            print('Set Exposure Time failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetEnumValue("GainAuto", 2)
        if ret != 0:
            print('Set Gain Auto failed! [{0:#X}]'.format(ret))

        return True

    def configure_slave_camera(self):
        # ch:设置触发模式为on | en:Set trigger mode as on
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0:
            raise Exception("set trigger mode fail! ret[0x%x]" % ret)

        # ch:设置触发选项为FrameBurstStart | en:Set trigger selector as FrameBurstStart
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSelector", 6)
        if ret != 0:
            raise Exception("set trigger selector fail! ret[0x%x]" % ret)

        # ch:设置触发源为Line0
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0)
        if ret != 0:
            print('Set Trigger Source failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerActivation", 0)
        if ret != 0:
            print('Set Trigger Activation failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", 50000)
        if ret != 0:
            print('Set Exposure Time failed! [{0:#X}]'.format(ret))

        ret = self.obj_cam.MV_CC_SetEnumValue("GainAuto", 2)
        if ret != 0:
            print('Set Gain Auto failed! [{0:#X}]'.format(ret))


    def configure_RGB_camera(self):
        # ch:设置触发模式为on | en:Set trigger mode as on
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        if ret != 0:
            raise Exception("set trigger mode fail! ret[0x%x]" % ret)

        # ch:设置触发选项为FrameBurstStart | en:Set trigger selector as FrameBurstStart
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSelector", 6)
        if ret != 0:
            raise Exception("set trigger selector fail! ret[0x%x]" % ret)

        # ch:设置触发源为Line0
        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0)
        if ret != 0:
            print('Set Trigger Source failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetEnumValue("TriggerActivation", 0)
        if ret != 0:
            print('Set Trigger Activation failed! [{0:#X}]'.format(ret))
            return False

        ret = self.obj_cam.MV_CC_SetEnumValue("ExposureAuto", 2)
        if ret != 0:
            print('Set Exposure Auto failed! [{0:#X}]'.format(ret))

        ret = self.obj_cam.MV_CC_SetEnumValue("GainAuto", 2)
        if ret != 0:
            print('Set Gain Auto failed! [{0:#X}]'.format(ret))

        return True

    # 开始取图
    def start_grabbing(self, n_index, queue):
        if not self.b_start_grabbing and self.b_open_device:
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.b_start_grabbing = False
                return ret
            self.b_start_grabbing = True
            print("start grabbing " + str(n_index) + "successfully!")
            try:
                self.work_thread(n_index, queue)
                self.b_thread_opened = True
            except TypeError:
                print('error: unable to start thread')
                self.b_start_grabbing = False
            return 0
        return MV_E_CALLORDER

    # 取图线程函数
    def work_thread(self, n_index, queue):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        while True:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:
                timestamp0 = time.time()  # 获取当前时间戳
                # 获取图像数据
                if self.buf_save_image_len < stOutFrame.stFrameInfo.nFrameLen:
                    self.buf_save_image = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                    self.buf_save_image_len = stOutFrame.stFrameInfo.nFrameLen

                cdll.msvcrt.memcpy(byref(self.st_frame_info), byref(stOutFrame.stFrameInfo),
                                   sizeof(MV_FRAME_OUT_INFO_EX))
                cdll.msvcrt.memcpy(byref(self.buf_save_image), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)

                # 将图像数据转换为 NumPy 数组
                image_data = np.ctypeslib.as_array(self.buf_save_image, shape=(self.st_frame_info.nFrameLen,))

                # 根据图像的像素格式进行转换
                if self.st_frame_info.enPixelType == PixelType_Gvsp_Mono8:
                    image = image_data.reshape((self.st_frame_info.nHeight, self.st_frame_info.nWidth))
                # elif self.st_frame_info.enPixelType == PixelType_Gvsp_RGB8_Packed:
                elif self.st_frame_info.enPixelType == PixelType_Gvsp_BayerGB8:
                    expected_length = self.st_frame_info.nHeight * self.st_frame_info.nWidth
                    if self.st_frame_info.nFrameLen == expected_length:
                        # 将 BayerGB8 数据转换为 OpenCV 格式
                        bayer_image = image_data.reshape((self.st_frame_info.nHeight, self.st_frame_info.nWidth))
                        # 使用 OpenCV 将 Bayer 格式转换为 RGB
                        image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GB2RGB)
                    else:
                        print(f"Error: Expected length {expected_length}, but got {self.st_frame_info.nFrameLen}")
                        continue
                else:
                    print(f"Unsupported pixel format:{self.st_frame_info.enPixelType}")
                    continue
                nDevTimeStamp = self.st_frame_info.nHostTimeStamp
                nDevTimeStamp = nDevTimeStamp/1000
                # print('nDevTimeStamp', n_index, nDevTimeStamp)

                # 将图像和相机索引放入队列
                if n_index == 1:
                    queue.put_single_image(image, "r", nDevTimeStamp)
                elif n_index == 2:
                    queue.put_single_image(image, "l", nDevTimeStamp)
                else:  # rgb相机
                    queue.put_image(image)
                    # cv2.imshow("RGB", image)
                    # key = cv2.waitKey(1)
                # 释放缓存
                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print("Camera[" + str(n_index) + "]: no data, ret = " + self.to_hex_str(ret))
                continue

class CameraInterface(object):
    def __init__(self, i):
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.cam_operation = None
        self.b_is_open = False
        self.b_is_grab = False
        self.b_is_trigger = False
        self.b_is_software_trigger = False
        self.initialize_SDK()
        self.i = i

    def ToHexStr(self, num):
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

    def initialize_SDK(self):
        MvCamera.MV_CC_Initialize()

    def enum_device(self):
        n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevicesEx2(n_layer_type, self.deviceList, '', SortMethod_SerialNumber)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            return
        if self.deviceList.nDeviceNum == 0:
            print("find no device!")
            return
        print("Find %d devices!" % self.deviceList.nDeviceNum)

        # 遍历设备列表并打印序列号
        if self.i < self.deviceList.nDeviceNum:
            device_info = cast(self.deviceList.pDeviceInfo[self.i], POINTER(MV_CC_DEVICE_INFO)).contents

            if device_info.nTLayerType == MV_GIGE_DEVICE:
                gige_info = device_info.SpecialInfo.stGigEInfo
                serial_number = ''.join(chr(gige_info.chSerialNumber[j]) for j in range(len(gige_info.chSerialNumber)) if
                                        gige_info.chSerialNumber[j] != 0)
                print("Device %d Serial Number: %s" % (self.i, serial_number))

            elif device_info.nTLayerType == MV_USB_DEVICE:
                usb_info = device_info.SpecialInfo.stUsb3VInfo
                serial_number = ''.join(chr(usb_info.chSerialNumber[j]) for j in range(len(usb_info.chSerialNumber)) if
                                        usb_info.chSerialNumber[j] != 0)
                print("Device %d Serial Number: %s" % (self.i, serial_number))

    def open_devices(self):
        if self.b_is_open:
            return

        print("deviceList.nDeviceNum", self.deviceList.nDeviceNum)
        n = self.deviceList.nDeviceNum
        if self.i < n:
            camObj = MvCamera()
            self.cam_operation = CameraOperation(camObj, self.deviceList, self.i)
            ret = self.cam_operation.open_device()
            if 0 != ret:
                print("open cam %d fail ret[0x%x]" % (self.i, ret))
                return
            else:
                # self.obj_cam_operation.append(cam_operation)
                self.b_is_open = True

        # 配置主相机和从相机
        if self.i == 2:
            self.cam_operation.configure_master_camera()  # 假设第二个相机是主相机
        if self.i == 1:
            self.cam_operation.configure_slave_camera()  # 假设第一个相机是从相机
        if self.i == 0:
            self.cam_operation.configure_RGB_camera()  # 第0个相机是RGB相机

    def start_grabbing(self, queue):
        if (not self.b_is_open) or self.b_is_grab:
            return

        if self.cam_operation is not None:
            ret = self.cam_operation.start_grabbing(self.i, queue)
            if 0 != ret:
                print('camera' + str(self.i) + ' start grabbing fail! ret = ' + self.ToHexStr(ret))
            self.b_is_grab = True

class ImageSetQueue(object):
    def __init__(self, queue_size, img_shape, tags, timestamp_max_deviation=0.01, timestamp_staple=1):
        super().__init__()
        self.queue_size = queue_size
        self.img_shape = img_shape
        self.vacant = multiprocessing.Queue()
        self.tags = tags
        self.timestamp_max_deviation = timestamp_max_deviation
        self.timestamp_staple = timestamp_staple
        self.get_timeout = 0.005  # block for 5ms to prevent racing
        self.put_timeout = 0.002  # block for 5ms to prevent racing
        for token in range(queue_size):
            self.vacant.put(token, block=True, timeout=self.put_timeout)
        self.occupied = multiprocessing.Queue()
        assert len(img_shape) == 2
        array_len = img_shape[0] * img_shape[1]
        array_type = 'B'  # unsigned char
        self.array = [multiprocessing.RawArray(array_type, array_len) for _ in range(queue_size)]

    def put_single_image(self, image, tag, nHostTimeStamp):
        timestamp = nHostTimeStamp
        try:
            token = self.vacant.get(block=True, timeout=self.get_timeout)
        except queue.Empty as e:
            raise queue.Empty(
                "unable to put image: vacant queue is empty, please consider increasing the queue length or call get_image more frequently") from e
        raw_data = image.tobytes()
        self.array[token][:] = raw_data[:]
        self.occupied.put((token, timestamp, tag), block=True, timeout=self.put_timeout)

    def get_image_set(self):
        def find(data_sorted, tags, deviation_max):
            data = data_sorted
            assert len(set(tags)) == len(tags)
            val_ind, tag_ind = 1, 2
            for begin in range(len(data)):
                best = {tag: None for tag in tags}
                choice = {tag: None for tag in tags}
                tag, val = data[begin][tag_ind], data[begin][val_ind]
                val_min = val - deviation_max
                choice[tag], best[tag] = begin, val
                for end in range(begin + 1, len(data)):
                    if data[end][val_ind] < val_min:
                        continue
                    tag, val = data[end][tag_ind], data[end][val_ind]
                    if best[tag] is None:
                        choice[tag], best[tag] = end, val
                    if all([choice[tag] is not None for tag in tags]):
                        return choice
            return None

        time_now = time.time()
        # print("estimated vacant/occupied queue size at the beginning", self.vacant.qsize(), self.occupied.qsize())
        images = []
        try:
            temp = 0
            while True:
                token, timestamp, tag = self.occupied.get(block=True, timeout=self.get_timeout)
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!12341')
                if timestamp < time_now - self.timestamp_staple:
                    self.vacant.put(token, block=True, timeout=self.put_timeout)
                    temp = temp + 1
                else:
                    images.append((token, timestamp, tag))
        except queue.Empty:
            # print("reached end of vacant")
            pass
        images = sorted(images, key=lambda x: x[1], reverse=True)  # sort by timestamp, reversed
        # print('number of staple/valid image', temp, len(images))
        # print('all images', images)
        choice = find(images, self.tags, self.timestamp_max_deviation)
        # print('selected', choice)
        if choice is None:
            for image in images:
                self.occupied.put(image, block=True, timeout=self.put_timeout)
            # print("estimated vacant/occupied queue size at the returning", self.vacant.qsize(), self.occupied.qsize())
            return None
        assert len(choice) == len(self.tags)
        tokens, timestamps, _ = zip(*[images[choice[tag]] for tag in self.tags])
        choice = [None for _ in self.tags]
        timestamps = {tag: timestamp for tag, timestamp in zip(self.tags, timestamps)}
        temp0 = 0
        temp1 = 0
        left_time = None
        for token, timestamp, tag in reversed(images):
            if token in tokens:  # 匹配上的两帧
                image = copy.deepcopy(np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.array[token]))
                self.vacant.put(token, block=True, timeout=self.put_timeout)
                choice[self.tags.index(tag)] = image
                if tag == 'l':
                    left_time = timestamp
            elif timestamp < timestamps[tag]:  # 比匹配上的同tag的还早的帧
                self.vacant.put(token, block=True, timeout=self.put_timeout)
                # self.vacant.put(token, block=True, timeout=0.001)
                temp0 += 1
            else:  # 其他的帧塞回
                self.occupied.put((token, timestamp, tag), block=True, timeout=self.put_timeout)
                temp1 += 1
        return choice, left_time

    def empty(self):
        return self.occupied.empty()

class ImageQueue(object):
    def __init__(self, queue_size, img_shape, timestamp_max_deviation=0.01, timestamp_staple=1):
        super().__init__()
        self.queue_size = queue_size
        self.img_shape = img_shape
        self.vacant = multiprocessing.Queue()
        self.timestamp_max_deviation = timestamp_max_deviation
        self.timestamp_staple = timestamp_staple
        self.get_timeout = 0.005  # block for 5ms to prevent racing
        self.put_timeout = 0.002  # block for 5ms to prevent racing
        for i in range(queue_size):
            self.vacant.put(i, block=True, timeout=self.put_timeout)
        self.occupied = multiprocessing.Queue()
        assert len(img_shape) == 3
        array_len = img_shape[0] * img_shape[1] * img_shape[2]
        array_type = 'B'  # unsigned char
        self.array = [multiprocessing.RawArray(array_type, array_len) for _ in range(queue_size)]

    def put_image(self, image):
        timestamp = time.time()
        if self.vacant.empty():
            index, _ = self.occupied.get(block=True, timeout=self.get_timeout)
        else:
            index = self.vacant.get(block=True, timeout=self.get_timeout)
        raw_data = image.tobytes()
        self.array[index][:] = raw_data[:]
        self.occupied.put((index, timestamp), block=True, timeout=self.put_timeout)

    def get_image_calibrate(self, left_time):
        def find(images, left_time, deviation_max):
            closest_image = min(images, key=lambda x: np.abs(x[1] - left_time).min())
            closest_token, closest_timestamp = closest_image
            if np.abs(closest_timestamp - left_time).min() <= deviation_max:
                return None
            else:
                return closest_image
        time_now = time.time()
        rgb_image = None
        closest_image = None
        images = []
        try:
            temp = 0
            while True:
                token, timestamp = self.occupied.get(block=True, timeout=self.get_timeout)
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!12341')
                if timestamp < time_now - self.timestamp_staple:
                    self.vacant.put(token, block=True, timeout=self.put_timeout)
                    temp = temp + 1
                else:
                    images.append((token, timestamp))
        except queue.Empty:
            # print("reached end of vacant")
            pass
        if len(images) > 0:
            images = sorted(images, key=lambda x: x[1], reverse=True)  # sort by timestamp, reversed
            closest_image = find(images, left_time, self.timestamp_max_deviation)

            if closest_image is None:
                for image in images:
                    self.occupied.put(image, block=True, timeout=self.put_timeout)
                    # print("estimated vacant/occupied queue size at the returning", self.vacant.qsize(), self.occupied.qsize())
                return None
            closest_token, closest_timestamp = closest_image
            for token, timestamp in reversed(images):
                if token == closest_token:
                    rgb_image = copy.deepcopy(np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.array[token]))
                    self.vacant.put(token, block=True, timeout=self.put_timeout)
                elif timestamp < closest_timestamp:
                    self.vacant.put(token, block=True, timeout=self.put_timeout)
                else:  # 其他的帧塞回
                    self.occupied.put((token, timestamp), block=True, timeout=self.put_timeout)
            return rgb_image
        else:
            return None

    def get_image(self):
        def find(images, left_time, deviation_max):
            closest_image = min(images, key=lambda x: np.abs(x[1] - left_time).min())
            closest_token, closest_timestamp = closest_image
            if np.abs(closest_timestamp - left_time).min() <= deviation_max:
                return None
            else:
                return closest_image
        time_now = time.time()
        rgb_image = None
        closest_image = None
        images = []
        try:
            temp = 0
            while True:
                token, timestamp = self.occupied.get(block=True, timeout=self.get_timeout)
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!12341')
                if timestamp < time_now - self.timestamp_staple:
                    self.vacant.put(token, block=True, timeout=self.put_timeout)
                    temp = temp + 1
                else:
                    images.append((token, timestamp))
                    # print(1)
        except queue.Empty:
            pass
        if len(images) > 0:
            images = sorted(images, key=lambda x: x[1], reverse=True)  # sort by timestamp, reversed
            closest_image = find(images, time_now, self.timestamp_max_deviation)
            closest_token, closest_timestamp = closest_image
            if closest_image is None:
                for image in images:
                    self.occupied.put(image, block=True, timeout=self.put_timeout)
                    # print("estimated vacant/occupied queue size at the returning", self.vacant.qsize(), self.occupied.qsize())
                return None
            for token, timestamp in reversed(images):
                if token == closest_token:
                    rgb_image = copy.deepcopy(np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.array[token]))
                    self.vacant.put(token, block=True, timeout=self.put_timeout)
                elif timestamp < closest_timestamp:
                    self.vacant.put(token, block=True, timeout=self.put_timeout)
                else:  # 其他的帧塞回
                    self.occupied.put((token, timestamp), block=True, timeout=self.put_timeout)
            return rgb_image
        else:
            return None

    def empty(self):
        return self.occupied.empty()

class ImageProcess:
    def __init__(self, datafile_r_to_l, datafile_rgb_to_l, real_world_points, img_shape):
        self.real_world_points = real_world_points
        self.target = None
        self.in_point = None
        self.is_input = None
        self.data = np.load(datafile_r_to_l)
        self.K_right = self.data["camera_matrix_r"]
        # print("K_right", self.K_right)
        self.K_left = self.data["camera_matrix_l"]
        # print("K_left", self.K_left)
        self.R, self.T = self.data["R"], self.data["T"]
        self.T_adjusted = self.T.flatten()  # 转换为一维数组
        # print("R", self.R)
        # print("T", self.T)
        # 畸变系数
        self.D_right = self.data["dist_coeffs_r"]
        self.D_left = self.data["dist_coeffs_l"]
        self.fx_right, self.fy_right, self.cx_right, self.cy_right = self.K_right[0, 0], self.K_right[1, 1], self.K_right[0, 2], self.K_right[1, 2]
        self.fx_left, self.fy_left, self.cx_left, self.cy_left = self.K_left[0, 0], self.K_left[1, 1], self.K_left[0, 2], self.K_left[1, 2]
        # 计算极线校正映射
        height_left, width_left = img_shape[0], img_shape[1]
        height_right, width_right = img_shape[0], img_shape[1]
        self.R1, self.R2, self.P1, self.P2, self.Q = self.data["R1"], self.data["R2"], self.data["P1"], self.data["P2"], self.data["Q"]
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(self.K_left, self.D_left, self.R1, self.P1, (width_left, height_left), cv2.CV_16SC2)
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(self.K_right, self.D_right, self.R2, self.P2, (width_right, height_right), cv2.CV_16SC2)

        self.data_rgb_to_l = np.load(datafile_rgb_to_l)
        self.R_l_to_rgb, self.T_l_to_rgb = self.data_rgb_to_l["R"], self.data_rgb_to_l["T"]
        self.T_l_to_rgb_adjusted = self.T_l_to_rgb.flatten()  # 转换为一维数组
        self.K_rgb = self.data_rgb_to_l["camera_matrix_rgb"]
        self.fx_rgb, self.fy_rgb, self.cx_rgb, self.cy_rgb = self.K_rgb[0, 0], self.K_rgb[1, 1], self.K_rgb[0, 2], self.K_rgb[1, 2]

    def find_seeds(self, thresholded_img, min_area=0, max_area=10000):
        """
        在阈值分割后的图像中找到所有高亮度区域的中心点作为种子点
        :param max_area: 最大面积
        :param min_area: 最小面积
        :param thresholded_img: 阈值分割后的二值图像
        :return: 高亮度区域的中心点列表
        """
        _, binary_img = cv2.threshold(thresholded_img, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seeds = []
        for contour in contours:
            area = cv2.contourArea(contour)  # 计算轮廓面积
            # print('area', area)
            if min_area <= area <= max_area:  # 只处理面积在指定范围内的轮廓
                # 计算轮廓的矩
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    # 计算轮廓的质心
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    seeds.append((cy, cx))  # 将 (x, y) 坐标转换为 OpenCV 的 (y, x) 坐标
        return seeds

    def region_growing(self, img, seeds, threshold=3):
        """
        区域生长算法实现
        :param img: 输入的灰度图像
        :param seeds: 种子点列表
        :param threshold: 像素值差异阈值，如果为None，则使用最大灰度值的一半
        :return: 分割后的二值图像
        """
        rows, cols = img.shape[:2]
        visited = np.zeros((rows, cols), dtype=bool)
        segmented = np.zeros((rows, cols), dtype=np.uint8)

        def grow_region(r, c, seed_value):
            stack = [(r, c)]
            while stack:
                r, c = stack.pop()
                if not (0 <= r < rows and 0 <= c < cols) or visited[r, c]:
                    continue
                visited[r, c] = True
                if abs(int(img[r, c]) - int(seed_value)) < threshold:
                    segmented[r, c] = 255
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                            stack.append((nr, nc))

        for seed in seeds:
            if not visited[seed]:  # 检查该种子是否已经被处理过
                grow_region(seed[0], seed[1], img[seed[0], seed[1]])

        return segmented

    def calculate_centroids(self, binary_image, original_image):
        """
        灰度质心算法计算二值图像中每个连通域的亚像素精度质心。
        :param binary_image: 二值图像 (numpy.ndarray)
        :param original_image: 原始图像 (numpy.ndarray)，用于计算亚像素精度质心
        :return: 质心列表 [(x, y), ...]
        """
        # 首先，对二值图像中的每个连通域进行标记
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        # 初始化一个空列表来存储每个连通域的质心
        subpixel_centroids = []
        for label in range(1, num_labels):  # 从1开始，因为0是背景
            # 获取当前连通域的掩码
            mask = (labels == label).astype(np.uint8)
            # 使用掩码从原始图像中提取当前连通域的灰度值
            gray_values = original_image * mask
            # 计算X和Y方向上的加权和
            x_coords, y_coords = np.meshgrid(np.arange(gray_values.shape[1]), np.arange(gray_values.shape[0]))
            total_intensity = gray_values.sum()
            if total_intensity > 0:
                centroid_x = (gray_values * x_coords).sum() / total_intensity
                centroid_y = (gray_values * y_coords).sum() / total_intensity

                # 添加质心到列表中
                subpixel_centroids.append((centroid_x, centroid_y))

        return subpixel_centroids

    def match_features(self, left_centroids, right_centroids, x_threshold=100, y_threshold=50):
        def undistort_points(points, K, D, R, P):
            points = np.expand_dims(points, axis=1)
            undistorted_points = cv2.undistortPoints(points, K, D, R=R, P=P)
            return np.atleast_2d(undistorted_points.squeeze())

        # 校正左右像素点
        # left_centroids_rectified = undistort_points(np.array(left_centroids, dtype=np.float32), self.K_left, self.D_left, self.R1, self.P1)
        # right_centroids_rectified = undistort_points(np.array(right_centroids, dtype=np.float32), self.K_right, self.D_right, self.R2, self.P2)

        # print("left_centroids_rectified", left_centroids_rectified)
        # print("right_centroids_rectified", right_centroids_rectified)
        right_centroids_rectified = right_centroids
        left_centroids_rectified = left_centroids
        matches = []

        for right_idx, right_point in enumerate(right_centroids_rectified):
            for left_idx, left_point in enumerate(left_centroids_rectified):
                # 计算X方向上的距离
                # x_distance = abs(left_point[0] - right_point[0])
                # 计算Y方向上的距离
                y_distance = abs(left_point[1] - right_point[1])

                # 检查是否满足条件
                if y_distance < y_threshold:
                    matches.append((left_centroids[left_idx], right_centroids[right_idx]))  # 添加原始未校正成对
        # print("matches_num", len(matches))
        return matches

    def calculate_3d_points(self, matches, right_frame, left_frame):
        calculated_3d_points = []
        points2d_3d = []
        def undistort_points(corners, camera_matrix, dist_coeffs):
            return cv2.undistortPoints(np.array(corners, dtype=np.float32), camera_matrix, dist_coeffs, None,
                                       camera_matrix)

        for left_center, right_center in matches:
            cX, cY = right_center
            cX_l, cY_l = left_center
            # print('right_center', right_center)
            # print('left_center', left_center)
            # 对图像中的点进行去畸变处理
            undistorted_l = undistort_points(left_center, self.K_left, self.D_left)
            undistorted_r = undistort_points(right_center, self.K_right, self.D_right)
            # print('undistorted_r', undistorted_r)
            # print('undistorted_l', undistorted_l)
            P_left = self.K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P_right = self.K_right @ np.hstack((self.R, self.T))
            point_4D = cv2.triangulatePoints(P_left, P_right, undistorted_l, undistorted_r)
            X = point_4D[:3] / point_4D[3]
            if X is not None:
                x, y, z = X.flatten()
                # 在图中标记三维坐标(左相机坐标系下)
                xyz_text = f"{x:.2f}, {y:.2f}, {z:.2f}"
                cX_int = int(cX)
                cY_int = int(cY)
                cX_l_int = int(cX_l)
                cY_l_int = int(cY_l)
                # print(type(calculated_3d_points))
                calculated_3d_points.append([x, y, z])
                points2d_3d.append([[x, y, z], [cX_l_int, cY_l_int]])
                cv2.putText(right_frame, xyz_text, (cX_int, cY_int + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
                cv2.putText(left_frame, xyz_text, (cX_l_int, cY_l_int + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
                # cv2.imshow('left_frame_with_contours', left_frame)
                # cv2.imshow('right_frame_with_contours', right_frame)
            else:
                print("No valid 3D point found for the given pair of points.")
        return calculated_3d_points, points2d_3d

    def cam2pix_rgb(self, xyz):
        """
        将相机坐标系下的点（xyz）转换为像素坐标系下的点（uv）。
        :param xyz: 相机坐标系下的三维点，形状为 (..., 3)
        :return: 像素坐标系下的二维点，形状为 (..., 2)
        """
        # 确保输入是 numpy 数组
        xyz = np.array(xyz)
        shape = xyz.shape
        # 检查最后一个维度是否为 3
        assert shape[-1] == 3, "最后一个维度必须为 3"

        # 将 xyz 转换为二维数组，形状为 (N, 3)
        xyz = xyz.reshape(-1, 3)
        x, y, z = xyz.T

        # 使用相机内参矩阵参数将三维点投影到二维图像平面
        u = x / z * self.fx_rgb + self.cx_rgb
        v = y / z * self.fy_rgb + self.cy_rgb
        uv = np.stack((u, v), axis=-1)

        # 恢复原始输入的形状
        uv = uv.reshape(*shape[:-1], 2)
        return uv

    def cam2pix_left(self, xyz):
        """
        将相机坐标系下的点（xyz）转换为像素坐标系下的点（uv）。
        :param xyz: 相机坐标系下的三维点，形状为 (..., 3)
        :return: 像素坐标系下的二维点，形状为 (..., 2)
        """
        # 确保输入是 numpy 数组
        xyz = np.array(xyz)
        shape = xyz.shape
        # 检查最后一个维度是否为 3
        assert shape[-1] == 3, "最后一个维度必须为 3"

        # 将 xyz 转换为二维数组，形状为 (N, 3)
        xyz = xyz.reshape(-1, 3)
        x, y, z = xyz.T

        # 使用相机内参矩阵参数将三维点投影到二维图像平面
        u = x / z * self.fx_left + self.cx_left
        v = y / z * self.fy_left + self.cy_left
        uv = np.stack((u, v), axis=-1)

        # 恢复原始输入的形状
        uv = uv.reshape(*shape[:-1], 2)
        return uv

    def cam2pix_right(self, xyz):
        """
        将相机坐标系下的点（xyz）转换为像素坐标系下的点（uv）。
        :param xyz: 相机坐标系下的三维点，形状为 (..., 3)
        :return: 像素坐标系下的二维点，形状为 (..., 2)
        """
        # 确保输入是 numpy 数组
        xyz = np.array(xyz)
        shape = xyz.shape
        # 检查最后一个维度是否为 3
        assert shape[-1] == 3, "最后一个维度必须为 3"

        # 将 xyz 转换为二维数组，形状为 (N, 3)
        xyz = xyz.reshape(-1, 3)
        x, y, z = xyz.T

        # 使用相机内参矩阵参数将三维点投影到二维图像平面
        u = x / z * self.fx_right + self.cx_right
        v = y / z * self.fy_right + self.cy_right
        uv = np.stack((u, v), axis=-1)

        # 恢复原始输入的形状
        uv = uv.reshape(*shape[:-1], 2)
        return uv

    def find_template(self, template, source, tol=0.05, max_error=0.05):
        source_distance_matrix = source[:, None, :] - source[None, :, :]
        source_distance_matrix = np.linalg.norm(source_distance_matrix, axis=-1)
        # upper triangular elements without diagonal
        source_distance_indices = np.triu_indices(source.shape[0], k=1)
        source_distance_vector = source_distance_matrix[source_distance_indices]

        template_distance_matrix = template[:, None, :] - template[None, :, :]
        template_distance_matrix = np.linalg.norm(template_distance_matrix, axis=-1)
        # upper triangular elements without diagonal
        template_distance_indices = np.triu_indices(template.shape[0], k=1)
        template_distance_vector = template_distance_matrix[template_distance_indices]

        distance_diff_matrix = template_distance_vector[:, None] - source_distance_vector[None, :]
        distance_match_matrix = np.absolute(distance_diff_matrix) <= tol

        solution_candidates = []  # vectors of solutions
        depth = len(template_distance_indices[0])
        nodes = [None] * template.shape[0]

        # replace all target to source
        # replace all template to target
        def dfs(edge):
            # print(nodes)
            # filling node connected to this edge
            if edge == depth:
                # done!
                if (None not in nodes) and (len(set(nodes)) == len(nodes)):
                    solution_candidates.append(nodes.copy())
                    # print("saved!")
                return
            template_index_i, template_index_j = [indices[edge] for indices in template_distance_indices]
            (nonzero,) = distance_match_matrix[edge].nonzero()
            for ind in nonzero:
                source_index_i, source_index_j = [indices[ind] for indices in source_distance_indices]
                # fill i to i, j to j
                check_i = (nodes[template_index_i] is None) or (nodes[template_index_i] == source_index_i)
                check_j = (nodes[template_index_j] is None) or (nodes[template_index_j] == source_index_j)
                if check_i and check_j:
                    restore_i = nodes[template_index_i]
                    restore_j = nodes[template_index_j]
                    nodes[template_index_i] = source_index_i
                    nodes[template_index_j] = source_index_j
                    dfs(edge + 1)
                    nodes[template_index_i] = restore_i
                    nodes[template_index_j] = restore_j
                # fill i to j, j to i
                check_i = (nodes[template_index_i] is None) or (nodes[template_index_i] == source_index_j)
                check_j = (nodes[template_index_j] is None) or (nodes[template_index_j] == source_index_i)
                if check_i and check_j:
                    restore_i = nodes[template_index_i]
                    restore_j = nodes[template_index_j]
                    nodes[template_index_i] = source_index_j
                    nodes[template_index_j] = source_index_i
                    dfs(edge + 1)
                    nodes[template_index_i] = restore_i
                    nodes[template_index_j] = restore_j

        dfs(0)
        # print("solution candidates", solution_candidates)

        def rigid_transform(A, B):
            """
            使用 Kabsch 算法计算刚性变换，使得 B ≈ R*A + t
            A, B: (n,dim) 点阵，必须一一对应
            返回 R (dim x dim) 和 t (dim,)
            """
            assert A.shape == B.shape, "两组点必须形状一致"
            centroid_A = A.mean(axis=0)
            centroid_B = B.mean(axis=0)
            AA = A - centroid_A
            BB = B - centroid_B
            H = AA.T @ BB
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            t = centroid_B - R @ centroid_A
            return R, t
        best_err = np.inf
        best_result = None
        for mapping in solution_candidates:
            candidate_pts = np.array([source[mapping[i]] for i in range(template.shape[0])])
            R, t = rigid_transform(template, candidate_pts)
            A_trans = (R @ template.T).T + t
            err = np.mean(np.linalg.norm(A_trans - candidate_pts, axis=1))
            if err < best_err and err < max_error:
                best_err = err
                best_result = (mapping, R, t, err)
                # 这里可直接返回第一个符合条件的映射
                return best_result
        if best_result is None:
            return None, None, None, None
        else:
            return best_result

    def calculate_affine_matrix(self, calculated_3d_points, points2d_3d, colored_image, left_frame_with_contours, needle_queue):
        # 确保 calculated_3d_points 是 numpy 数组并且是 float32 类型
        calculated_3d_points = np.array(calculated_3d_points, dtype=np.float32)
        # 确保 real_world_points 是 numpy 数组并且是 float32 类型
        real_world_points = np.array(self.real_world_points, dtype=np.float32)
        # print("calculated_3d_points_num", calculated_3d_points.shape[0])
        # print("real_world_points_num", real_world_points.shape[0])
        ordered_points_3d = None

        # 使用函数在 U 中寻找模板 A 的对应点及刚性变换
        mapping, R_est, t_est, err = self.find_template(template=real_world_points, source=calculated_3d_points, tol=1, max_error=15)
        if mapping is not None:
            ordered_points_3d = np.array([calculated_3d_points[mapping[i]] for i in range(real_world_points.shape[0])])
            ordered_points2d_3d = [points2d_3d[mapping[i]] for i in range(real_world_points.shape[0])]
            # print("找到对应关系（模板索引 -> U 中对应索引）：", mapping)
            # print("估计的旋转矩阵 R:\n", R_est)
            # print("估计的平移向量 t:", t_est)
            # print("平均匹配误差：", err)
        # else:
             # print("未能找到满足条件的对应关系。")

        tip_mapped_point = None
        assist_mapped_point = None
        mapped_point_uv_left = None
        # 检查点的数量是否一致且不少于4个点
        if (mapping is not None) and (ordered_points_3d.shape == real_world_points.shape):
            # print(11111111)
            '''
            correspondence, error = self.find_correspondence(real_world_points, calculated_3d_points)
            if correspondence is not None:
                ordered_points_3d = calculated_3d_points[list(correspondence)]
                ordered_points2d_3d = [points2d_3d[i] for i in list(correspondence)]
            '''
            for i in range(4):
                # print(i)
                cv2.putText(colored_image, str(i),
                            (ordered_points2d_3d[i][1][0], ordered_points2d_3d[i][1][1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 2)
                line2d = [ordered_points2d_3d[i - 1][1], ordered_points2d_3d[i][1]]
                line2d_center = int((line2d[0][0] + line2d[1][0]) / 2), int((line2d[0][1] + line2d[1][1]) / 2)
                line3d = [ordered_points2d_3d[i - 1][0], ordered_points2d_3d[i][0]]
                line3d_length = np.sum((np.array(line3d[0]) - np.array(line3d[1])) ** 2) ** 0.5
                line3d_length = np.linalg.norm(np.array(line3d[0]) - np.array(line3d[1]))
                cv2.putText(colored_image, str(i),
                            (ordered_points2d_3d[i][1][0], ordered_points2d_3d[i][1][1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 2)
                cv2.putText(colored_image, f"{line3d_length:.1f}", (line2d_center[0], line2d_center[1] + 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

            '''
            ordered_points_3d_r = np.dot(ordered_points_3d, self.R.T) + self.T_adjusted
            retval, affine_matrix, inliers = cv2.estimateAffine3D(real_world_points, ordered_points_3d_r)
            if retval:
                transform, scale = cv2.estimateAffine3D(src=real_world_points, dst=ordered_points_3d_r,
                                                        force_rotation=True)
                # decompose(transform)
                # print("scale", scale)
                assert transform.shape == (3, 4)
                transform[:, 0] = transform[:, 0] * scale
                transform[:, 1] = transform[:, 1] * scale
                transform[:, 2] = transform[:, 2] * scale
                new_row = np.array([[0, 0, 0, 1]])  # 定义要添加的行
                transform = np.vstack((transform, new_row))  # 将新行添加到矩阵t的末尾
                tip_point = np.array([298, 0, -13], dtype=np.float32)
                assist_point = np.array([-1000, -1000, -1000], dtype=np.float32)
                # tip_point = np.array([0, 0, 18], dtype=np.float32)
                # 使用仿射变换矩阵将目标点映射到三维坐标系
                tip_point_homogeneous = np.append(tip_point, 1)  # 转换为齐次坐标
                tip_mapped_point = np.dot(transform, tip_point_homogeneous)
                tip_mapped_point = tip_mapped_point[:3]
                # print("tip_point", tip_mapped_point)
                # print("ordered_points_3d", ordered_points_3d)
                assist_point_homogeneous = np.append(assist_point, 1)
                assist_mapped_point = np.dot(transform, assist_point_homogeneous)
                assist_mapped_point = assist_mapped_point[:3]
                # print("assist_point", assist_mapped_point)
                # 将三维点映射到二维图像平面
                mapped_point_uv = self.cam2pix_right(tip_mapped_point)
                mapped_point_uv_left = np.rint(mapped_point_uv).astype(int)
            '''

            # 将坐标转到rgb坐标系下
            # print("left_3d", ordered_points_3d)
            ordered_points_3d = np.dot(ordered_points_3d, self.R_l_to_rgb.T) + self.T_l_to_rgb_adjusted
            # print("rgb_3d", ordered_points_3d)
            # 计算仿射变换矩阵
            retval, affine_matrix, inliers = cv2.estimateAffine3D(real_world_points, ordered_points_3d)
            if retval:
                transform, scale = cv2.estimateAffine3D(src=real_world_points, dst=ordered_points_3d, force_rotation=True)
                # decompose(transform)
                # print("scale", scale)
                assert transform.shape == (3, 4)
                transform[:, 0] = transform[:, 0] * scale
                transform[:, 1] = transform[:, 1] * scale
                transform[:, 2] = transform[:, 2] * scale
                new_row = np.array([[0, 0, 0, 1]])  # 定义要添加的行
                transform = np.vstack((transform, new_row))  # 将新行添加到矩阵t的末尾

                assist_point1 = np.array([20.508523, -14.29687, 25], dtype=np.float32)
                assist_point = np.array([64.224432, 22.157896, 25], dtype=np.float32)
                # 使用仿射变换矩阵将目标点映射到三维坐标系
                assist_point1_homogeneous = np.append(assist_point1, 1)  # 转换为齐次坐标
                assist1_mapped_point = np.dot(transform, assist_point1_homogeneous)
                assist1_mapped_point = assist1_mapped_point[:3]
                # print("tip_point", tip_mapped_point)
                # print("ordered_points_3d", ordered_points_3d)
                assist_point_homogeneous = np.append(assist_point, 1)
                assist_mapped_point = np.dot(transform, assist_point_homogeneous)
                assist_mapped_point = assist_mapped_point[:3]
                # print("assist_point", assist_mapped_point)

                # 将三维点映射到二维图像平面
                tip_point = np.array([298, 0, 5], dtype=np.float32)
                assist_point = np.array([0, 0, 5], dtype=np.float32)
                tip_point = np.array([-2.366469, -47.110506, 5], dtype=np.float32)
                tip_point_homogeneous = np.append(tip_point, 1)  # 转换为齐次坐标
                tip_mapped_point = np.dot(transform, tip_point_homogeneous)
                tip_mapped_point = tip_mapped_point[:3]
                mapped_points = np.vstack((assist1_mapped_point, assist_mapped_point, tip_mapped_point))
                mapped_point_uv = self.cam2pix_rgb(tip_mapped_point)
                mapped_point_uv = np.rint(mapped_point_uv).astype(int)

                # print("mapped_point_uv", mapped_point_uv)
                needle_dict = {'transform': transform, 'ordered_points_3d': ordered_points_3d, 'mapped_points': mapped_points, 'tip_point_uv': mapped_point_uv}
                needle_queue.put(needle_dict)
                # cX, cY = mapped_point_uv
                # 在RGB图中标记二维坐标（暂时有误）
                # cv2.circle(left_frame_with_contours, mapped_point_uv, 5, (0, 255, 0), -1)
                # cv2.putText(left_frame_with_contours, f"({cX}, {cY})", (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 输出映射后的三维坐标
                # print("Mapped 3D coordinates:", tip_mapped_point)
            else:
                print("Affine transformation could not be estimated.")
        # else:
            # print("Insufficient or mismatched points for affine transformation.")
        return colored_image, tip_mapped_point, assist_mapped_point, mapped_point_uv_left

class vtkTimerCallback(object):
    def __init__(self, needle_queue, rgb_queue, renderer, actor_needle, actor_sphere1, actor_sphere2, actor_sphere3, actor_sphere4, actor_sphere1_1, actor_sphere2_1, actor_sphere3_1, actor_sphere4_1, actor_tip, actor_in,
                 target_actor, road_actor, actor_needle_cylinder, iren):
        self.needle_queue = needle_queue
        self.rgb_queue = rgb_queue
        self.renderer = renderer
        self.actor_needle = actor_needle
        self.actor_sphere1 = actor_sphere1
        self.actor_sphere2 = actor_sphere2
        self.actor_sphere3 = actor_sphere3
        self.actor_sphere4 = actor_sphere4
        self.actor_sphere1_1 = actor_sphere1_1
        self.actor_sphere2_1 = actor_sphere2_1
        self.actor_sphere3_1 = actor_sphere3_1
        self.actor_sphere4_1 = actor_sphere4_1
        self.actor_tip = actor_tip
        self.actor_in = actor_in
        self.target_actor = target_actor
        self.road_actor = road_actor
        self.actor_needle_cylinder = actor_needle_cylinder
        self.iren = iren
        self.timerId = None
        self.assist_points_and_tip_point = None
        self.in_y_point = None
        self.target_input = None
        self.window_size = 3
        self.buffer = []
        self.buffer_0 = []
        self.buffer_1 = []
        self.buffer_2 = []
        self.buffer_3 = []
        self.data_rgb_to_l = np.load("rgb_to_left_calib_params.npz")
        self.K_rgb = self.data_rgb_to_l['camera_matrix_rgb']
        self.D_rgb = self.data_rgb_to_l['dist_coeffs_rgb']
        self.update_flag = False
        self.dict = None
        self.target_point, self.in_point, self.deepth, self.ordered_points = self.calculate_transform()

    def execute(self, obj, event):
        if not self.rgb_queue.empty():
            t, tip_point_uv, ordered_points_3d, virtual_tip = self.update_matrix()
            matrix = vtk.vtkMatrix4x4()
            image = self.rgb_queue.get_image()
            if image is not None:
                # 获取图像的尺寸
                h, w = image.shape[:2]
                # 计算新的相机矩阵和ROI（Region of Interest）
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.K_rgb, self.D_rgb, (w, h), 1, (w, h))
                # 使用计算得到的新相机矩阵进行去畸变
                undistorted_image = cv2.undistort(image, self.K_rgb, self.D_rgb, None, new_camera_matrix)
                # 可选：裁剪掉黑色区域
                x, y, w, h = roi
                # undistorted_image = undistorted_image[y:y + h, x:x + w]
                # 显示原始图像和去畸变后的图像
                # cv2.imshow('Original Image', image)
                # cv2.imshow('Undistorted Image', undistorted_image)
                if tip_point_uv is not None:
                    # cv2.circle(image, tip_point_uv, 5, (0, 255, 0), -1)
                    cv2.circle(undistorted_image, tip_point_uv, 5, (0, 255, 0), -1)

                # 调用函数提取中间最大的矩形部分
                # middle_rectangle = self.extract_middle_rectangle(undistorted_image)
                # self.update_image(middle_rectangle)
                self.update_image(undistorted_image)
                h, w, channels = undistorted_image.shape  # 获取原始图像的尺寸
                extra_h = 300  # 定义额外空间的高度
                new_h = h + extra_h  # 创建一个更大的图像
                new_image = np.zeros((new_h, w, 3), dtype=np.uint8)
                new_image[:h, :] = undistorted_image  # 将原始图像粘贴到新图像的顶部
                # 在新图像的下方区域填充淡蓝色背景
                science_blue = (204, 102, 0)  # 淡蓝色 BGR 值
                new_image[h:, :] = science_blue

                new_image = self.add_text_to_corner(new_image, position='bottom_right')
                self.draw_circle_and_point_rotation(new_image, self.in_point, self.target_point, self.in_y_point, self.assist_points_and_tip_point, virtual_tip)
                # if self.in_point is not None and distance is not None:
                    # if math.isnan(distance) is not True:
                        # self.add_value(self.buffer, round(distance, 1))
                    # average_distance = self.get_average(self.buffer)
                    # print('average_distance', average_distance)
                    # new_image = self.add_text_to_corner(new_image, average_distance, position='bottom_right')
                    # 在新图像左下画圆
                    # self.draw_circle_and_point(new_image, self.in_point, self.target_point, self.in_y_point, self.assist_points_and_tip_point)
                '''
                if self.ordered_points is not None:
                    if ordered_points_3d is not None:
                        distance_0 = self.calculate_distance(self.ordered_points[0], ordered_points_3d[0])
                        if math.isnan(distance_0) is not True:
                            self.add_value(self.buffer_0, round(distance_0, 1))
                        average_distance_0 = self.get_average(self.buffer_0)
                        new_image = self.add_text_to_corner(new_image, average_distance_0, position='bottom_right_0')
                        distance_1 = self.calculate_distance(self.ordered_points[1], ordered_points_3d[1])
                        if math.isnan(distance_1) is not True:
                            self.add_value(self.buffer_1, round(distance_1, 1))
                        average_distance_1 = self.get_average(self.buffer_1)
                        new_image = self.add_text_to_corner(new_image, average_distance_1, position='bottom_right_1')
                        distance_2 = self.calculate_distance(self.ordered_points[2], ordered_points_3d[2])
                        if math.isnan(distance_2) is not True:
                            self.add_value(self.buffer_2, round(distance_2, 1))
                        average_distance_2 = self.get_average(self.buffer_0)
                        new_image = self.add_text_to_corner(new_image, average_distance_2, position='bottom_right_2')
                        distance_3 = self.calculate_distance(self.ordered_points[3], ordered_points_3d[3])
                        if math.isnan(distance_3) is not True:
                            self.add_value(self.buffer_3, round(distance_3, 1))
                        average_distance_3 = self.get_average(self.buffer_3)
                        new_image = self.add_text_to_corner(new_image, average_distance_3, position='bottom_right_3')
                '''
                cv2.imshow('RGB', new_image)
                key = cv2.waitKey(1)
                if key == ord('s'):
                    self.update_flag = not self.update_flag
                if key == ord('c'):
                    print(self.assist_points_and_tip_point[2])

                iren = obj
                iren.GetRenderWindow().Render()
            if t is None:
                return
            t = t.flatten()
            matrix.DeepCopy(t)
            self.actor_needle.SetUserMatrix(matrix)
            # self.actor_needle.SetVisibility(True)
            iren = obj
            iren.GetRenderWindow().Render()

    def update_matrix(self):
        transform = None
        tip_point_uv = None
        distance = None
        virtual_tip = None
        dict = None
        while not self.needle_queue.empty():
            dict = self.needle_queue.get()
            self.dict = dict
        if dict is None:
            dict = self.dict
        if dict is None:
            return None, None, None, None
        transform = dict.get('transform')
        ordered_points_3d = dict.get('ordered_points_3d')
        mapped_points = dict.get('mapped_points')
        tip_point_uv = dict.get('tip_point_uv')
        if transform is not None:
            self.assist_points_and_tip_point = mapped_points
            virtual_tip = self.point_on_ray(A=mapped_points[0], B=mapped_points[1], distance=50)
            self.actor_tip.SetPosition(*virtual_tip)
            # distance = self.distance_point_to_line(virtual_tip, self.target_point, self.in_point)
            self.update_cylinder_endpoints(self.actor_needle_cylinder, virtual_tip, mapped_points[0])
            if self.update_flag:
                self.ordered_points = ordered_points_3d
            self.actor_sphere1.SetPosition(*self.ordered_points[0])
            self.actor_sphere2.SetPosition(*self.ordered_points[1])
            self.actor_sphere3.SetPosition(*self.ordered_points[2])
            self.actor_sphere4.SetPosition(*self.ordered_points[3])
            self.actor_sphere1_1.SetPosition(*ordered_points_3d[0])
            self.actor_sphere2_1.SetPosition(*ordered_points_3d[1])
            self.actor_sphere3_1.SetPosition(*ordered_points_3d[2])
            self.actor_sphere4_1.SetPosition(*ordered_points_3d[3])

        target_c = dict.get('target_c')
        target_input = dict.get('target_input')
        assist = dict.get('assist_point')
        in_point = dict.get('in_point')
        if target_input is not None:
            x, y, z = in_point
            y += 100000
            self.in_y_point = (x, y, z)
            self.target_input = target_input
            self.in_point = in_point
            self.target_actor.SetPosition(*target_input)
            self.actor_in.SetPosition(*in_point)
            self.update_cylinder_endpoints(self.road_actor, in_point, target_input)
        elif target_c is not None:
            self.target_actor.SetPosition(*target_c)
            self.update_cylinder_endpoints(self.road_actor, target_c, assist)
        elif self.in_point is not None:
            self.target_actor.SetPosition(*self.target_point)
            self.actor_in.SetPosition(*self.in_point)
            C = self.point_on_ray(A=self.target_point, B=self.in_point, distance=150)
            self.update_cylinder_endpoints(self.road_actor, C, self.target_point)
        if transform is None:
            return None, None, None, None
        return transform, tip_point_uv, ordered_points_3d, virtual_tip

    def update_cylinder_endpoints(self, actor, new_position1, new_position2):
        mapper = actor.GetMapper()
        input_connection = mapper.GetInputConnection(0, 0)
        tube_filter = input_connection.GetProducer()
        line_source_connection = tube_filter.GetInputConnection(0, 0)
        line_source = line_source_connection.GetProducer()
        line_source.SetPoint1(*new_position1)
        line_source.SetPoint2(*new_position2)
        line_source.Update()
        tube_filter.Update()

    def update_image(self, frame):
        # 更新 VTK 图像
        if frame is not None:
            # 获取图像的高度和宽度
            height, width = frame.shape[:2]
            # 确定中心点
            center_x, center_y = width // 2, height // 2
            # 计算矩形的宽度和高度
            min_dim = min(width, height)
            rect_size = min_dim // 2  # 中间最大的矩形大小
            # 裁剪图像
            frame = frame[
                    center_y - rect_size:center_y + rect_size,
                    center_x - rect_size:center_x + rect_size
                    ]
            frame = frame[..., ::-1]  # 将BRG转换为RGB
            vtkImage = vtk.vtkImageData()
            vtkImage.SetDimensions(frame.shape[1], frame.shape[0], 1)
            vtkImage.SetNumberOfScalarComponents(frame.shape[2], vtkImage.GetInformation())
            arr = frame
            pd = vtkImage.GetPointData()
            new_arr = arr[::-1].reshape((-1, arr.shape[2]))
            pd.SetScalars(numpy_to_vtk(new_arr))
            texture = vtk.vtkTexture()
            # texture.SetInputConnection(jpeg_reader.GetOutputPort())
            texture.SetInputData(vtkImage)
            texture.Update()
            self.renderer.SetBackgroundTexture(texture)
            self.renderer.SetTexturedBackground(True)

    def calculate_distance(self, point1, point2):
        if point1 is None or point2 is None:
            return math.nan
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        distance = round(distance, 1)
        return distance

    def distance_point_to_line(self, A, B, C):
        # 将点转换为numpy数组
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)

        # 计算向量AB和AC
        AB = B - A
        AC = C - A

        # 计算叉积AC x AB
        cross_product = np.cross(AC, AB)

        # 计算叉积的模长
        numerator = np.linalg.norm(cross_product)

        # 计算向量AB的模长
        denominator = np.linalg.norm(AB)

        # 计算距离
        distance = numerator / denominator

        return distance

    def add_value(self, buffer, value):
        buffer.append(value)
        if len(buffer) > self.window_size:
            buffer.pop(0)

    def get_average(self, buffer):
        if not buffer:
            return 0
        return round(sum(buffer) / len(buffer), 1)

    def add_text_to_corner(self, image, position='bottom_right'):
        distance = self.deepth
        h, w, _ = image.shape  # 获取图像尺寸
        font_path = "SimHei.ttf"
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 确定起始位置
        if position == 'bottom_right':
            font_size = 96  # 设置字体大小
            # 使用PIL来添加中文文本
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(font_path, font_size)  # 加载字体
            # 固定文本部分
            fixed_part = "入针深度: "
            variable_part = " mm"
            # 计算固定部分的宽度和高度
            _, _, fixed_width, _ = draw.textbbox((0, 0), fixed_part, font=font)
            # 计算变动部分的宽度和高度
            _, _, variable_width, variable_height = draw.textbbox((0, 0), str(distance) + variable_part, font=font)
            text_x = w - fixed_width - variable_width - 10
            text_y = h - variable_height - 10
            # 绘制固定部分
            draw.text((text_x+400, text_y-100), fixed_part, font=font, fill=(255, 255, 255))
            # 计算变动部分的位置
            distance_text_x = text_x + fixed_width
            draw.text((distance_text_x, text_y), str(distance) + " mm", font=font, fill=(255, 255, 255))
        elif position == 'bottom_right_0':
            font_size = 48  # 设置字体大小
            # 使用PIL来添加中文文本
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(font_path, font_size)  # 加载字体
            fixed_part = "0号距离: "
            variable_part = " mm"
            _, _, fixed_width, _ = draw.textbbox((0, 0), fixed_part, font=font)
            _, _, variable_width, variable_height = draw.textbbox((0, 0), str(distance) + variable_part, font=font)
            text_x = w - fixed_width - variable_width - 10
            text_y = h - variable_height - 260  # 10
            # 绘制固定部分
            draw.text((text_x, text_y), fixed_part, font=font, fill=(255, 255, 255))
            # 计算变动部分的位置
            distance_text_x = text_x + fixed_width
            draw.text((distance_text_x, text_y), str(distance) + " mm", font=font, fill=(255, 255, 255))
        elif position == 'bottom_right_1':
            font_size = 48  # 设置字体大小
            # 使用PIL来添加中文文本
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(font_path, font_size)  # 加载字体
            fixed_part = "1号距离: "
            variable_part = " mm"
            _, _, fixed_width, _ = draw.textbbox((0, 0), fixed_part, font=font)
            _, _, variable_width, variable_height = draw.textbbox((0, 0), str(distance) + variable_part, font=font)
            text_x = w - fixed_width - variable_width - 10
            text_y = h - variable_height - 177  # 10
            # 绘制固定部分
            draw.text((text_x, text_y), fixed_part, font=font, fill=(255, 255, 255))
            # 计算变动部分的位置
            distance_text_x = text_x + fixed_width
            draw.text((distance_text_x, text_y), str(distance) + " mm", font=font, fill=(255, 255, 255))
        elif position == 'bottom_right_2':
            font_size = 48  # 设置字体大小
            # 使用PIL来添加中文文本
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(font_path, font_size)  # 加载字体
            fixed_part = "2号距离: "
            variable_part = " mm"
            _, _, fixed_width, _ = draw.textbbox((0, 0), fixed_part, font=font)
            _, _, variable_width, variable_height = draw.textbbox((0, 0), str(distance) + variable_part, font=font)
            text_x = w - fixed_width - variable_width - 10
            text_y = h - variable_height - 93  # 10
            # 绘制固定部分
            draw.text((text_x, text_y), fixed_part, font=font, fill=(255, 255, 255))
            # 计算变动部分的位置
            distance_text_x = text_x + fixed_width
            draw.text((distance_text_x, text_y), str(distance) + " mm", font=font, fill=(255, 255, 255))
        elif position == 'bottom_right_3':
            font_size = 48  # 设置字体大小
            # 使用PIL来添加中文文本
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(font_path, font_size)  # 加载字体
            fixed_part = "3号距离: "
            variable_part = " mm"
            _, _, fixed_width, _ = draw.textbbox((0, 0), fixed_part, font=font)
            _, _, variable_width, variable_height = draw.textbbox((0, 0), str(distance) + variable_part, font=font)
            text_x = w - fixed_width - variable_width - 10
            text_y = h - variable_height - 10  # 10
            # 绘制固定部分
            draw.text((text_x, text_y), fixed_part, font=font, fill=(255, 255, 255))
            # 计算变动部分的位置
            distance_text_x = text_x + fixed_width
            draw.text((distance_text_x, text_y), str(distance) + " mm", font=font, fill=(255, 255, 255))
        else:
            raise ValueError("Unsupported position: {}".format(position))
        font_size = 46  # 设置字体大小
        # 使用PIL来添加中文文本
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(font_path, font_size)  # 加载字体
        txt = "旋转"
        draw.text((70, h - 280), txt, font=font, fill=(255, 255, 255))
        txt = "平移"
        draw.text((int(w/2)-50, h - 280), txt, font=font, fill=(255, 255, 255))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式
        return image

    def calculate_transform(self):
        # 给定点集
        # 相机坐标系
        body_points = np.array([
            [-98.06854245, 202.91547903, 951.70421836],
            [129.6827553, 213.67490032, 958.57602901],
            [-100.44236143, 208.30480044, 1069.16058497],
            [ 127.50806453, 217.10933224, 1073.69711968]
        ], dtype=np.float32)
        # Oc = np.array([-240.83324343, 4.97080309, -5.08946637], dtype=np.float32)
        data = np.load("rgb_to_left_calib_params.npz")
        R, T = data["R"], data["T"]
        Oc = T.flatten()
        # ct坐标系
        body_ct_points = np.array([
            [435, 182, 273],
            [95, 201, 283],
            [435, 179, 41],
            [91, 196, 53]
        ], dtype=np.float32)
        '''
        body_ct_points = np.array([
            [-95, 95, 20],
            [-95, -95, 20],
            [95, 95, 20],
            [95, -95, 20]
        ], dtype=np.float32)'''
        target_points = np.array([
            [0, 0, 10],  # 中
            [0, -50, 10],  # 下
            [-50, 0, 20],  # 左
            [0, 50, 30],  # 上
            [50, 0, 40]  # 右
        ], dtype=np.float32)
        # 器械坐标系
        needle_points = np.array([
            [0, 0, 18],
            [40.448864, 44.315792, 18],
            [88, 0, 18],
            [41.017045, -28.59374, 18]
        ], dtype=np.float32)
        tip_point = np.array([64.224432, 22.157896, 25], dtype=np.float32)
        assist_point = np.array([20.508523, -14.29687, 25], dtype=np.float32)
        assist_point1 = np.array([64.224432, 22.157896, 24], dtype=np.float32)
        # 计算直线的方向向量
        direction_vector = assist_point - tip_point
        # 找到垂直于该直线的一个方向向量（可以使用叉积）
        # 我们需要一个辅助向量来计算叉积，这里选择一个简单的辅助向量 [0, 0, 1]
        auxiliary_vector = np.array([0, 0, 1], dtype=np.float32)
        perpendicular_vector = np.cross(direction_vector, auxiliary_vector)
        # 单位化垂直方向向量
        perpendicular_unit_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
        # 计算目标点，距离tip_point为1
        distance = 1
        assist_point2 = tip_point + perpendicular_unit_vector * distance

        transform, scale = cv2.estimateAffine3D(body_ct_points, body_points, force_rotation=True)
        assert transform.shape == (3, 4)
        transform[:, 0] = transform[:, 0] * scale
        transform[:, 1] = transform[:, 1] * scale
        transform[:, 2] = transform[:, 2] * scale

        # 定义target_point和in_point
        ct_data = np.load("ct_data.npz")
        target_point = ct_data["target_point"]
        # target_point = target_points[0]
        x, y, z = target_point
        z = z - 15
        target_point = np.array([x, y, z], dtype=np.float32)
        in_point = ct_data["in_point"]
        # in_point = np.array([x, y, z], dtype=np.float32)
        x, y, z = in_point
        z = z - 15
        in_point = np.array([x, y, z], dtype=np.float32)
        x, y, z = in_point
        x = x + 10
        in_y_point = np.array([x, y, z], dtype=np.float32)

        # 将点转换为齐次坐标
        in_point_homogeneous = np.append(in_point, 1).reshape((4, 1))
        target_point_homogeneous = np.append(target_point, 1).reshape((4, 1))
        in_y_point_homogeneous = np.append(in_y_point, 1).reshape((4, 1))

        # 应用变换矩阵
        in_point_camera = np.dot(transform, in_point_homogeneous)[:3].flatten()
        target_point_camera = np.dot(transform, target_point_homogeneous)[:3].flatten()
        self.in_y_point = np.dot(transform, in_y_point_homogeneous)[:3].flatten()

        # 计算射线方向向量
        direction_vector = in_point_camera - target_point_camera
        direction_vector /= np.linalg.norm(direction_vector)

        # 计算tip_point_camera的坐标（距in_point_camera 200）
        distance_tip_in = 50
        tip_point_camera = in_point_camera + direction_vector * distance_tip_in

        # 计算tip_point和assist_point在器械坐标系中的距离
        distance_tip_assist_tool = np.linalg.norm(tip_point - assist_point)

        # 计算assist_point_camera的坐标（距tip_point_camera相同距离）
        assist_point_camera = tip_point_camera + direction_vector * distance_tip_assist_tool

        # 计算从Oc到target_point_camera的向量
        vector_Oc_target = target_point_camera - Oc

        # 计算vector_Oc_target在垂直于direction_vector方向上的分量
        perpendicular_component = vector_Oc_target - np.dot(vector_Oc_target, direction_vector) * direction_vector

        # 单位化垂直分量
        perpendicular_unit_vector = perpendicular_component / np.linalg.norm(perpendicular_component)

        assist_point_camera1 = tip_point_camera + perpendicular_unit_vector

        # 第三个维度的单位向量
        axis3 = np.cross(direction_vector, perpendicular_unit_vector)
        axis3 /= np.linalg.norm(axis3)

        assist_point_camera2 = tip_point_camera + axis3

        # 器械坐标系中的三点及其对应的相机坐标系中的点
        tool_points = np.array([tip_point, assist_point, assist_point1, assist_point2], dtype=np.float32)
        camera_points = np.array([tip_point_camera, assist_point_camera, assist_point_camera1, assist_point_camera2], dtype=np.float32)
        tool_points = np.array([tip_point, assist_point, assist_point1], dtype=np.float32)
        camera_points = np.array([tip_point_camera, assist_point_camera, assist_point_camera1],
                                 dtype=np.float32)

        # 使用三对对应的点计算新的变换矩阵
        new_transform, new_scale = cv2.estimateAffine3D(tool_points, camera_points, force_rotation=True)

        # 确保变换矩阵形状正确并应用缩放因子
        assert new_transform.shape == (3, 4)
        new_transform[:, 0] = new_transform[:, 0] * new_scale
        new_transform[:, 1] = new_transform[:, 1] * new_scale
        new_transform[:, 2] = new_transform[:, 2] * new_scale

        # 将needle_points转换到相机坐标系下
        needle_points_camera = []
        for point in needle_points:
            point_homogeneous = np.append(point, 1).reshape((4, 1))
            point_camera = np.dot(new_transform, point_homogeneous)[:3].flatten()
            needle_points_camera.append(point_camera)

        needle_points_camera = np.array(needle_points_camera, dtype=np.float32)

        try:
            deepth = ct_data["deepth"]
        except:
            deepth = self.calculate_distance(target_point_camera, in_point_camera)

        return target_point_camera, in_point_camera, round(deepth, 1), needle_points_camera

    def point_on_ray(self, A, B, distance):
        # Convert points to numpy arrays for vector operations
        A = np.array(A)
        B = np.array(B)

        # Calculate the direction vector AB
        AB = B - A

        # Normalize the direction vector AB
        unit_AB = AB / np.linalg.norm(AB)

        # Scale the unit vector by the given distance
        BC = unit_AB * distance

        # Calculate point C
        C = B + BC

        return C

    def draw_circle_and_point_rotation(self, image, in_xyz, tumor_xyz, assist_xyz, tips_xyz, virtual_tip):
        if tips_xyz is None:
            # print(11111111)
            return None
        # 将列表转换为 NumPy 数组
        tumor_xyz = np.array(tumor_xyz)
        in_xyz = np.array(in_xyz)
        assist_xyz = np.array(assist_xyz)
        # 计算 z 轴
        z = tumor_xyz - in_xyz
        z_hat = z / np.linalg.norm(z)
        # print("z_hat", z_hat)
        # 计算 y 轴
        a = assist_xyz - in_xyz
        # print("a", a)
        a_parallel = (np.dot(a, z_hat)) * z_hat
        a_perpendicular = a - a_parallel
        norm = np.linalg.norm(a_perpendicular)
        if norm > 1e-10:  # 设定一个合适的阈值，用于判断向量是否有效
            y_hat = a_perpendicular / norm
        else:
            # 处理零向量的情况，可以抛出异常、打印警告或设置默认值
            # print("Warning: Attempted to normalize a zero vector.")
            y_hat = np.zeros_like(a_perpendicular)  # 或者采取其他适当的行动
        # 计算 x 轴
        x_hat = np.cross(y_hat, z_hat)
        # 计算射线 l
        l = tips_xyz[0] - in_xyz
        # 计算射线 l 与 z 轴的夹角
        cos_theta = np.dot(l, z_hat) / (np.linalg.norm(l) * np.linalg.norm(z_hat))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_l_z = np.degrees(theta)
        # 调整 angle_l_z 使其在 0 到 90 度之间
        if cos_theta < 0:
            angle_l_z = 180 - angle_l_z
        # 计算射线 l 在 xy 平面的投影
        l_parallel = (np.dot(l, z_hat)) * z_hat
        l_perpendicular = l - l_parallel
        # 计算射线 l 在 xy 平面的投影与 x 轴的夹角
        angle_l_xy = np.degrees(np.arctan2(np.dot(l_perpendicular, y_hat), np.dot(l_perpendicular, x_hat)))
        if math.isnan(angle_l_xy):
            return None
        h, w, _ = image.shape  # 获取图像尺寸1324 1280
        radius = 100
        # 将角度转换为弧度
        angle_l_z_rad = np.radians(angle_l_z)  # 限制 angle_l_z 最大为 45 度
        angle_l_xy_rad = np.radians(angle_l_xy)
        # print(angle_l_xy)
        # 计算点 A 的半径
        r = radius * (min(angle_l_z, 15) / 15)  # 点 A 从圆心到圆弧的距
        center = (120, h - 120)
        # 计算点 A 的位置
        x = int(center[0] - r * np.cos(angle_l_xy_rad))
        y = int(center[1] - r * np.sin(angle_l_xy_rad))
        # print(x,y)
        # 绘制圆和点
        cv2.circle(image, center, radius, (255, 255, 255), 5)
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

        # 画第二个圆
        # 计算向量 AB 和 AC
        AB = tumor_xyz - in_xyz
        AC = virtual_tip - in_xyz
        # 计算叉积 AC x AB
        cross_product = np.cross(AC, AB)
        # 计算叉积的模长和向量 AB 的模长
        magnitude_cross_product = np.linalg.norm(cross_product)
        magnitude_AB = np.linalg.norm(AB)
        # 计算点到直线的距离
        distance = magnitude_cross_product / magnitude_AB
        radius = 100
        r = radius * (min(distance, 20)/20)
        center = (int(w/2), h - 120)
        # 计算投影系数 t
        t = np.dot(AC, AB) / np.dot(AB, AB)
        # 计算垂足点 D 的坐标
        D = in_xyz + t * AB
        # 计算向量 CD，即从 virtual_tip 到垂足点的向量
        CD = virtual_tip - D
        angle_CD_xy = np.degrees(np.arctan2(np.dot(CD, y_hat), np.dot(CD, x_hat)))
        angle_CD_xy_rad = np.radians(angle_CD_xy)
        x = int(center[0] - r * np.cos(angle_CD_xy_rad))
        y = int(center[1] - r * np.sin(angle_CD_xy_rad))
        # 绘制圆和点
        cv2.circle(image, center, radius, (255, 255, 255), 5)
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)


    def extract_middle_rectangle(self, image):
        # 获取图像的高度和宽度
        height, width = image.shape[:2]
        # 确定中心点
        center_x, center_y = width // 2, height // 2
        # 计算矩形的宽度和高度
        min_dim = min(width, height)
        rect_size = min_dim // 2  # 中间最大的矩形大小
        # 裁剪图像
        cropped_image = image[
                        center_y - rect_size:center_y + rect_size,
                        center_x - rect_size:center_x + rect_size
                        ]
        return cropped_image

class Show_vtk(object):
    def __init__(self, needle_queue, rgb_queue, real_world_points):
        self.real_world_points = real_world_points
        self.needle_queue = needle_queue if needle_queue is not None else []
        self.rgb_queue = rgb_queue
        colors = vtkNamedColors()

        self.actor_needle = self.create_needle_actor(filename="光标版.STL")
        self.actor_sphere1 = self.create_sphere_actor([-1000, -1000, -1000])
        self.actor_sphere1.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.actor_sphere2 = self.create_sphere_actor([-1000, -1000, -1000])
        self.actor_sphere2.GetProperty().SetColor(0.0, 0.0, 1.0)
        self.actor_sphere3 = self.create_sphere_actor([-1000, -1000, -1000])
        self.actor_sphere3.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.actor_sphere4 = self.create_sphere_actor([-1000, -1000, -1000])

        self.actor_sphere1_1 = self.create_sphere_actor([-1000, -1000, -1000])
        self.actor_sphere1_1.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.actor_sphere2_1 = self.create_sphere_actor([-1000, -1000, -1000])
        self.actor_sphere2_1.GetProperty().SetColor(0.0, 0.0, 1.0)
        self.actor_sphere3_1 = self.create_sphere_actor([-1000, -1000, -1000])
        self.actor_sphere3_1.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.actor_sphere4_1 = self.create_sphere_actor([-1000, -1000, -1000])

        self.actor_tip = self.create_sphere_actor([-1000, -1000, -1000])
        self.actor_in = self.create_sphere_actor([-1000, -1000, -1000])
        self.actor_in.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.target_actor = self.create_sphere_actor([0, 0, -200])
        self.target_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.road_actor = self.create_cylinder_actor([0, 0, -100], [0, 0, -100])
        self.actor_needle_cylinder = self.create_cylinder_actor([0, 0, -100], [0, 0, -100])
        self.actor_needle_cylinder.GetProperty().SetColor(0.0, 0.0, 1.0)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(colors.GetColor3d("SteelBlue"))
        self.renderer.AddActor(self.actor_needle)
        self.renderer.AddActor(self.actor_sphere1)
        self.renderer.AddActor(self.actor_sphere2)
        self.renderer.AddActor(self.actor_sphere3)
        self.renderer.AddActor(self.actor_sphere4)
        self.renderer.AddActor(self.actor_sphere1_1)
        self.renderer.AddActor(self.actor_sphere2_1)
        self.renderer.AddActor(self.actor_sphere3_1)
        self.renderer.AddActor(self.actor_sphere4_1)
        self.renderer.AddActor(self.actor_tip)
        self.renderer.AddActor(self.actor_in)
        self.renderer.AddActor(self.target_actor)
        self.renderer.AddActor(self.road_actor)
        self.renderer.AddActor(self.actor_needle_cylinder)
        self.actor_needle.SetVisibility(False)
        # self.actor_tip.SetVisibility(False)
        if True:
            # 隐藏actor
            self.actor_sphere1.SetVisibility(False)
            self.actor_sphere2.SetVisibility(False)
            self.actor_sphere3.SetVisibility(False)
            self.actor_sphere4.SetVisibility(False)
            self.actor_sphere1_1.SetVisibility(False)
            self.actor_sphere2_1.SetVisibility(False)
            self.actor_sphere3_1.SetVisibility(False)
            self.actor_sphere4_1.SetVisibility(False)

        camera = vtk.vtkCamera()
        camera.SetPosition(0, 0, 0)  # 240.83324343, -4.97080309, 5.08946637)# 相机位于原点
        camera.SetFocalPoint(0, 0, 1)  # 相机看向Z轴正方向
        camera.SetViewUp(0, -1, 0)  # 设置相机的向上向量为Y轴正方向，确保正确的旋转
        camera.SetViewAngle(34.474748893543)  # 设置相机的视场角
        # camera.SetViewAngle(38)
        self.renderer.SetActiveCamera(camera)

        renderWindow = vtkRenderWindow()
        renderWindow.AddRenderer(self.renderer)
        # 设置渲染窗口的大小（宽度为800像素，高度为600像素）
        renderWindow.SetSize(600, 600)

        self.renderWindowInteractor = vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(renderWindow)
        self.renderWindowInteractor.Initialize()
        cb = vtkTimerCallback(self.needle_queue, self.rgb_queue, self.renderer, self.actor_needle, self.actor_sphere1,
                              self.actor_sphere2, self.actor_sphere3, self.actor_sphere4, self.actor_sphere1_1,
                              self.actor_sphere2_1, self.actor_sphere3_1, self.actor_sphere4_1, self.actor_tip, self.actor_in, self.target_actor,
                              self.road_actor, self.actor_needle_cylinder, self.renderWindowInteractor)
        self.renderWindowInteractor.AddObserver("TimerEvent", cb.execute)
        cb.timerId = self.renderWindowInteractor.CreateRepeatingTimer(10)

        renderWindow.Render()
        self.renderWindowInteractor.Start()

    def create_needle_actor(self, filename):
        # 创建一个STL读取器并读取STL文件
        stl_reader = vtk.vtkSTLReader()
        stl_reader.SetFileName(filename)
        stl_reader.Update()

        # 创建一个PolyDataMapper并将读取器的输出设置为输入
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(stl_reader.GetOutputPort())

        # 创建一个Actor并将Mapper设置为输入
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor

    def create_sphere_actor(self, position):
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(10)
        sphere.SetThetaResolution(30)
        sphere.SetPhiResolution(30)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        sphere_actor.SetPosition(*position)
        return sphere_actor

    def create_cylinder_actor(self, position1, position2):
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(*position1)
        lineSource.SetPoint2(*position2)
        tubeFilter = vtk.vtkTubeFilter()
        tubeFilter.SetInputConnection(lineSource.GetOutputPort())
        tubeFilter.SetRadius(3)
        tubeFilter.SetNumberOfSides(5)
        tubeFilter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tubeFilter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        return actor

class PointInputApp:
    def __init__(self, root, needle_queue):
        self.root = root
        self.root.title("Point Input")
        self.needle_queue = needle_queue
        self.target_point = None
        self.last_target_point = None
        self.in_point = None

        # 创建标签和输入框
        self.create_widgets()

    def create_widgets(self):
        # Target Point
        self.target_label = tk.Label(self.root, text="Target Point (x, y, z):")
        self.target_label.grid(row=0, column=0, padx=10, pady=5)

        self.target_x = tk.Entry(self.root)
        self.target_x.grid(row=0, column=1, padx=5, pady=5)
        self.target_y = tk.Entry(self.root)
        self.target_y.grid(row=0, column=2, padx=5, pady=5)
        self.target_z = tk.Entry(self.root)
        self.target_z.grid(row=0, column=3, padx=5, pady=5)

        # In Point
        self.in_label = tk.Label(self.root, text="In Point (x, y, z):")
        self.in_label.grid(row=1, column=0, padx=10, pady=5)

        self.in_x = tk.Entry(self.root)
        self.in_x.grid(row=1, column=1, padx=5, pady=5)
        self.in_y = tk.Entry(self.root)
        self.in_y.grid(row=1, column=2, padx=5, pady=5)
        self.in_z = tk.Entry(self.root)
        self.in_z.grid(row=1, column=3, padx=5, pady=5)

        # Confirm Button
        self.confirm_button = tk.Button(self.root, text="Confirm", command=self.save_points)
        self.confirm_button.grid(row=2, column=0, columnspan=4, pady=10)

    def save_points(self):
        try:
            target_point = np.array([
                float(self.target_x.get()),
                float(self.target_y.get()),
                float(self.target_z.get())
            ])
            in_point = np.array([
                float(self.in_x.get()),
                float(self.in_y.get()),
                float(self.in_z.get())
            ])

            # 保存点坐标
            self.target_point = target_point
            self.in_point = in_point
            needle_dict = {'target_input': self.target_point, 'in_point': self.in_point}
            self.needle_queue.put(needle_dict)

            # 显示保存成功消息
            messagebox.showinfo("Success", f"Points saved:\nTarget Point: {target_point}\nIn Point: {in_point}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values for all coordinates.")