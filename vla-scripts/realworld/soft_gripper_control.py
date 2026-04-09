"""新的控制夹爪的sdk,需要更新robotic_arm_controller这个类里面的控制夹爪的函数(datacollect里面的都不要多线程控制夹爪了哈哈哈)"""

import time

import crcmod
import serial
from tqdm import *


class ControlRoot:
    def __init__(self, com="/dev/ttyUSB0"):
        self.sc = serial.Serial(port=com, baudrate=115200)
        self.crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)

    # def calCrc(self, array):
    #     print('wanfang',array)
    #     bytes_ = b''
    #     for i in range(array.__len__()):
    #         bytes_ = bytes_ + array[i].to_bytes(1, byteorder='big', signed=True)
    #     crc = hex(self.crc16(bytes_))
    #     crcQ = '0x' + crc[-2] + crc[-1]
    #     crcH = '0x' + crc[-4] + crc[-3]
    #     return int(crcQ.encode(), 16), int(crcH.encode(), 16)

    def calCrc(self, array):
        bytes_ = b""
        for i in range(array.__len__()):
            bytes_ = bytes_ + array[i].to_bytes(1, byteorder="big", signed=True)
        crc = self.crc16(bytes_).to_bytes(2, byteorder="big", signed=False)
        crcH = int.from_bytes(crc[0:1], byteorder="big", signed=False)
        crcQ = int.from_bytes(crc[1:2], byteorder="big", signed=False)
        return crcQ, crcH

    def readSerial(self):
        # BTime = time.time()
        time.sleep(0.008)
        readContent = self.sc.read_all()
        return readContent

    def sendCmd(self, ModbusHighAddress, ModbusLowAddress, Value=0x01, isSet=True, isReadSerial=True):
        if isSet:
            SetAddress = 0x06
        else:
            SetAddress = 0x03
        Value = Value if Value >= 0 else Value - 1
        bytes_ = Value.to_bytes(2, byteorder="big", signed=True)
        ValueHexQ = int.from_bytes(bytes_[0:1], byteorder="big", signed=True)
        ValueHexH = int.from_bytes(bytes_[1:2], byteorder="big", signed=True)
        array = [0x01, SetAddress, ModbusHighAddress, ModbusLowAddress, ValueHexQ, ValueHexH]
        currentValueQ, currentValueH = self.calCrc(array)
        setValueCmd = [
            0x01,
            SetAddress,
            ModbusHighAddress,
            ModbusLowAddress,
            ValueHexQ,
            ValueHexH,
            currentValueQ,
            currentValueH,
        ]
        for i in range(setValueCmd.__len__()):
            setValueCmd[i] = setValueCmd[i] if setValueCmd[i] >= 0 else setValueCmd[i] + 256
        self.sc.write(setValueCmd)

        if isReadSerial:
            back = self.readSerial()
            value = int.from_bytes(back[3:5], byteorder="big", signed=True)
            if value < 0:
                value = value + 1
            self.sc.flush()
            return value
        time.sleep(0.005)
        return None


def isRange(value, min_, max_):
    if not min_ <= value <= max_:
        raise RuntimeError("Out of range")


class DHgripper:
    def __init__(self, ControlInstance=ControlRoot()):
        self.Hand = ControlInstance
        self.HandInit()
        self.InitFeedback()

    # 初始化夹爪
    def HandInit(self):
        self.Hand.sendCmd(ModbusHighAddress=0x01, ModbusLowAddress=0x00, Value=1)

    # 力值
    def Force(self, value):
        isRange(value, 20, 100)
        self.Hand.sendCmd(ModbusHighAddress=0x01, ModbusLowAddress=0x01, Value=value)

    # 位置
    def Position(self, value):
        isRange(value, 0, 1000)
        self.Hand.sendCmd(ModbusHighAddress=0x01, ModbusLowAddress=0x03, Value=value)

    # 速度
    def Velocity(self, value):
        isRange(value, 0, 1000)
        self.Hand.sendCmd(ModbusHighAddress=0x01, ModbusLowAddress=0x04, Value=value)

    # 绝对旋转角度
    def AbsoluteRotate(self, cmd):
        isRange(cmd, -32768, 32767)
        self.Hand.sendCmd(ModbusHighAddress=0x01, ModbusLowAddress=0x05, Value=cmd, isReadSerial=False)

    # 旋转速度
    def RotateVelocity(self, value):
        isRange(value, 1, 100)
        self.Hand.sendCmd(ModbusHighAddress=0x01, ModbusLowAddress=0x07, Value=value, isReadSerial=False)

    # 旋转力值
    def RotateForce(self, value):
        isRange(value, 20, 100)
        self.Hand.sendCmd(ModbusHighAddress=0x01, ModbusLowAddress=0x08, Value=value)

    # 相对旋转角度
    def RelativeRotate(self, cmd):
        isRange(cmd, -32768, 32767)
        self.Hand.sendCmd(ModbusHighAddress=0x01, ModbusLowAddress=0x09, Value=cmd)

    def InitFeedback(self):
        back = self.Hand.sendCmd(ModbusHighAddress=0x02, ModbusLowAddress=0x00, isSet=False)
        while back == 0:
            self.HandInit()
            back = self.Hand.sendCmd(ModbusHighAddress=0x02, ModbusLowAddress=0x00, isSet=False)
            print(back)
        while back == 2:
            time.sleep(0.1)
            back = self.Hand.sendCmd(ModbusHighAddress=0x02, ModbusLowAddress=0x00, isSet=False)
            print(back)


class ReadStatus:
    def __init__(self, ControlInstance=ControlRoot()):
        self.Hand = ControlInstance

    def RTRotateAngle(self):
        back = self.Hand.sendCmd(ModbusHighAddress=0x02, ModbusLowAddress=0x08, isSet=False)
        print(back)
