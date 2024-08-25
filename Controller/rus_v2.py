from asyncio import sleep
from operator import pos
from datetime import datetime
import sys
import struct
import warnings
from matplotlib import pyplot as plt
import os
from pathlib import Path

import serial
import serial.tools.list_ports
import time
import threading

from functools import partial
from matplotlib.backend_bases import Event

import numpy as np
import math
import csv

import matplotlib
from numpy.core.arrayprint import str_format
from numpy.lib.type_check import imag, real

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter, FuncFormatter
from matplotlib.widgets import Cursor, MultiCursor
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

from PyQt5.uic import loadUiType
from PyQt5.QtCore import QRegExp, QTimer, QSettings, QDir, Qt, QObject, QElapsedTimer
from PyQt5.QtGui import QRegExpValidator, QPalette, QColor, QBitmap, QPixmap
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QMainWindow, QMessageBox, QDialog, QFileDialog, QPushButton, \
    QLabel, QSpinBox
from PyQt5.QtNetwork import QAbstractSocket, QTcpSocket
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplashScreen
# np.set_printoptions(threshold=sys.maxsize)
import time

import socket

'''
Input format to left bitshift number by 60
'''
def inpf(num):
    print(num*(2**60))
    return num*(2**60)

def hex_to_dec(str):
    '''Flip the order'''
    chunks = [str[i - 2:i] for i in range(len(str), 0, -2)]
    decimals = [chunk for chunk in chunks]
    print(decimals)
    # chunks = [str[i:i + 2] for i in range(0, len(str), 2)]
    # decimals = [chunk for chunk in chunks]
    # print(decimals)
    '''convert to decimal'''
    decimals = [int(chunk, 16) for chunk in chunks]
    print(decimals)
    '''convert to binary'''
    binary = ["{0:08b}".format(int(chunk, 16))  for chunk in chunks]
    binary = ''.join(binary)
    print(binary)
    # print(len(binary))
    # case = ''.join(binary.split())[:-60]
    # print(f'case: {case}')
    return binary

def binary_to_dec(str):
    decimal = 0
    for i in range(len(str)):
        digit = int(str[i])
        power = len(str) - 1 - i
        decimal += digit * (2 ** power)
    return decimal

def check_case(str):
    # Convert hexadecimal string to bytes
    byte_str = bytes(hex_str)
    print(byte_str)
    # Convert each byte to binary representation
    binary_list = [bin(byte)[2:].zfill(8) for byte in byte_str]
    # Concatenate binary representations
    binary_str = ''.join(binary_list)
    print(binary_str)
    print(len(binary_str))
    case = ''.join(binary_str.split())[:-60]
    print(f'case: {case}')
    return case

'''
ctrl_handler:

case 0 (reset)
command & 0xFFFF = 1111111111111111 (16 digits)

case 1 (CMD_IDN)
command does not appear. checks IDN status

case 2 (CMD_STOP)
offset = command & 0xFFFFFFFF; = 11111111111111111111111111111111 (32 digits)

case 3 (get status)
offset = command & 0xFFFFFFFF; = 11111111111111111111111111111111 (32 digits)

case 4 (get temperature)
we dont need this 

case 5 (CMD_CONNECT)

case 6 (arm; wtf is arm)
samples = command & 0xFFFFFFFF; 32

case 7 (software trigger)
command does not appear

case 8 (read from start of buffer)
samples = command & 0xFFFFFFFF; 32

case 9 (read data chunk)
start_offset= command & 0x3FFFFFFF; = 00111111111111111111111111111111 (32 digits)
end_offset = (command  >> 30)& 0x3FFFFFFF; 

case 11 (get config)
offset = command & 0xFFFFFFFF; 32

case 12: //set config
offset = (command >> 32)& 0xFF; = 11111111 (8 digit)
status=command & 0xFFFFFFFF; 32 

case 13: // read RX FIFO
samples = command & 0xFFFFFFFF; 32

case 14: //read calibration

case 15: //Write TX FIFO
samples = command & 0xFFFFFFFF; 32
'''

def bytestring(hex_str):
    byte_str = bytes(hex_str)
    print(byte_str)
    return byte_str

if __name__ == "__main__":
    # num = "2c010000000000d0"
    # hex_to_dec(num)

    # ip.addr == 10.84.241.73
    print(socket.gethostname())

    HOST = "10.84.241.73" 
    PORT = 1002 
    address = (HOST, PORT)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        s.connect(address)
        
        arg = struct.pack("<Q", 2**60+2**62)
        s.sendall(arg)
        print(arg)
        arg = struct.pack("<Q", 2**63+2**62+2**61)
        s.sendall(arg)
        print(arg)
        hex_to_dec("2f0000001c0000c0")
        arg = struct.pack("<Q", binary_to_dec("1100000000000000000000000001110000000000000000000000000000101111")) #fix this
        s.sendall(arg)
        print(arg)       
        response = s.recv(1024)
        print("response", response) 
    finally:
        s.close


    # HOST = "10.84.241.73" 
    # PORT = 1002 
    # socket = QTcpSocket()
    # socket.connectToHost(HOST, PORT)

    # print(socket.waitForConnected())

    # '''
    # Trial and Error notes;
    # 5/23 -
    # Input tests:
    # struct.pack("Q", 2**60) returns Qbytearray b'\x0b\xb0g\x00'
    # '''
    # socket.write(struct.pack("<Q", inpf(1)))
    # # socket.write(struct.pack("<Q", inpf(5)))

    # socket.waitForReadyRead()
    # print(socket.readAll())