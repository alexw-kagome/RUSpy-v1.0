# Control program for the Red Pitaya RUS experiment
# Copyright (C) 2024  Alexander Won
# Based on the data acquisition code from Albert Migliori and Red Pitaya vector network analyzer from Pavel Demin
#
# Ref: Reviews of Scientific Instruments Vol.90 Issue 12 Dec 2019 pgs. 121401 ff.
# Copyright (C) 2022  Albert Migliori
# Based heavily on the Control program for the Red Pitaya vector network analyzer
# Copyright (C) 2021  Pavel Demin

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
import copy

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

'''
Measurement Class to store data
'''
class Measurement:
    def __init__(self, start, stop, size):
        self.freq = np.linspace(start, stop, size)
        self.data = np.zeros(size, np.complex64)

class rus(QObject):
    def __init__(self):
        super(rus, self).__init__()

        self.initialize()
        
        '''
        TCP socket to communicate with RP.
        '''
        self.socket = QTcpSocket(self)
        self.socket.readyRead.connect(self.read_data)
        # self.socket.readyRead.connect(self.rp_read)
        self.socket.connected.connect(self.connect)
        self.socket.error.connect(self.error)

        '''
        startTimer: Detect timeout during connection
        alexSweepTimer: Measures time took for experiment (Alex edition)
        '''
        self.startTimer = QTimer(self)
        self.startTimer.timeout.connect(self.timeout)

        self.alexSweepTimer = QElapsedTimer()
    
    def initialize(self):
        self.idle = True
        self.reading = False
        self.sweep_start = 0
        self.sweep_stop = 0
        self.sweep_size = 0
        self.stepValue = 0
        self.addressValue = ""
        self.socketError = ""
        self.connect_status = ""
        self.elapsedTime = 0
        self.rate = 0 
        self.voltage = 0
        self.count = 0

        
        # buffer and offset for the incoming samples
        self.buffer = bytearray(16 * 32768)
        self.offset = 0
        self.data = np.frombuffer(self.buffer, np.complex64)

        # create measurements
        self.reim = Measurement(self.sweep_start, self.sweep_stop, self.sweep_size)
    
    '''
    RP is connected.
    '''
    def connect(self):
        self.startTimer.stop()
        # self.initialize()
        self.connect_status = "connected"
        self.idle = False
        self.set_corr(0)
        self.set_phase1(0)
        self.set_phase2(0)
        self.set_volt2(0)
        self.set_gpio(1)

    '''
    RP timed out; update status to timeout.
    '''
    def timeout(self):
        self.connect_status = "timeout"

    '''
    RP throws error during connection.
    '''
    def error(self):
        self.startTimer.stop()
        print("error")
        self.connect_status = self.socket.errorString()

    '''
    connect to RP.
    '''
    def start(self, address):
        if self.idle:
            self.socket.connectToHost(address, 1001)
            self.startTimer.start(2000)
        else:
            self.stop()
    
    '''
    stop RP.
    '''
    def stop(self):
        self.idle = True
        self.cancel()
        self.socket.abort()
        self.connect_status = ""

    '''
    cancel sweep.
    '''
    def cancel(self):
        self.reading = False
        self.socket.write(struct.pack("<I", 11 << 28))

    '''
    read data from RP.
    read_data is called whenever socket is readyRead.
    note: we only retreive data once experiment is over.
    '''
    def read_data(self):
        if self.count == 0:
            print(time.time() - self.start_time)
        self.count = self.count + 1
        while (self.socket.bytesAvailable() > 0):
            if not self.reading:
                self.socket.readAll()
                return
            size = self.socket.bytesAvailable()
            # print(size)
            limit = 16 * self.sweep_size
            # collect data
            if self.offset + size < limit:
                self.buffer[self.offset:self.offset + size] = self.socket.read(size)
                self.offset += size
            # we are at the end of buffer; sweep over
            else:
                self.buffer[self.offset:limit] = self.socket.read(limit - self.offset)
                adc1 = self.data[0::2]
                # adc2 = self.data[1::2]
                attr = getattr(self, "reim")
                start = self.sweep_start
                stop = self.sweep_stop
                size = self.sweep_size
                attr.freq = np.linspace(start, stop, size)
                attr.data = adc1[0:size].copy()
                attr.unity = np.full(size, 1)
                self.reading = False
                self.elapsedTime = self.alexSweepTimer.elapsed()/1000
                print(self.elapsedTime)
                # print(self.reim.freq)
                # print(self.reim.data)

    """
    Performs Sweep by input commands.

    General note about passing in arguments for RP: self.socket.write(struct.pack("<I", VALUE1 << 28 | VALUE2)))

    a<<b is binary bit shift left. You shift a to b times left.
    Example:
    3<<5 = 110000 since 3 = 11
    16383 = 11111111111111 (14 1s)

    red pitaya has 14 bit ADC resolution, 2^14.
    | is bitwise OR operator. So the first (right) 14 digits become the value that we are passing in (VALUE2) and the last 14 digits (left) 
    are the left shifted number which I suspect to be the address to indicate what parameter I am setting (VALUE1).
    "<I" is formatting
    """
    def sweep(self):
        print("sweep")
        self.start_time = time.time()
        if self.idle: return
        self.offset = 0
        self.count = 0
        self.reading = True
        # print(f"{self.sweep_start} {self.sweep_stop} {self.sweep_size}")
        self.socket.write(struct.pack("<I", 0 << 28 | int(self.sweep_start * 1000)))
        self.socket.write(struct.pack("<I", 1 << 28 | int(self.sweep_stop * 1000)))
        self.socket.write(struct.pack("<I", 2 << 28 | int(self.sweep_size)))
        self.socket.write(struct.pack("<I", 10 << 28))
        # print('sweep data inputted')
        self.alexSweepTimer.start()

    """
    Sets the rate of data collection. Higher the rate, more data points are collected and averaged out.
    There is also a quantity points per second which goes like [1500, 500, 150, 50]. These correspond to the rate [30, 100, 300, 1000].
    points per second is roughly the same as (sweep size) / (total sweep time) so its meaning is how many frequency points does it scan per second.
    Rate is then how long it takes to scan one frequency point.
    So the longer we scan, higher the rate, more data points are averaged out, lower the noise.
    rate of 300 have similar sweep time as the labview software.
    """
    def set_rate(self, value):

        if self.idle: return
        rate = [30, 100, 300, 1000, 3000, 10000, 30000, 100000][int(value)]
        self.rate = rate
        # print(f"{rate}")
        # rate = [10, 50, 100, 500, 1000, 5000, 10000, 50000][value]
        # self.rateValue.addItems(["5000", "1000", "500", "100", "50", "10", "5", "1"])
        self.socket.write(struct.pack("<I", 3 << 28 | int(rate)))
        # print('3<<')

    """
    set it to 0. Not needed for RUS. VNA artifact
    """
    def set_corr(self, value):

        if self.idle: return
        self.socket.write(struct.pack("<I", 4 << 28 | int(value & 0xfffffff)))
        # print('4<<')

    """
    set it to 0. Not needed for RUS. VNA artifact
    """
    def set_phase1(self, value):
        if self.idle: return
        self.socket.write(struct.pack("<I", 5 << 28 | int(value)))
        # print('5<<')

    """
    set it to 0. Not needed for RUS. VNA artifact
    """
    def set_phase2(self, value):
        if self.idle: return
        self.socket.write(struct.pack("<I", 6 << 28 | int(value)))
        # print('6<<')

    """
    Sets Voltage Level.
    If we want x volts, we need to input 16383*x (16384 = 2^14 - 1).  
    https://docs.python.org/3/library/struct.html
    https://doc.qt.io/qt-6/qiodevice.html#write
    """
    def set_volt1(self, value):
        if self.idle: return
        # print(f"{value}")
        data = 0 if (value <=0 or value > 2) else int(16383*value)   
        self.voltage = data/16383
        self.socket.write(struct.pack('<I', 7<<28 | int(data)))
        # print('7<<')

    """
    set it to 0. Not needed for RUS. VNA artifact
    """
    def set_volt2(self, value):
        if self.idle: return
        data = 0 if (value <=0 or value >2) else int(16383*value)
        self.socket.write(struct.pack("<I", 8 << 28 | int(data)))
        # print('8<<')

    """
    set it to 1. Not needed for RUS. VNA artifact
    """
    def set_gpio(self, value):
        if self.idle: return
        self.socket.write(struct.pack("<I", 9 << 28 | int(value)))
        # print('9<<')

    def set_idle(self, value):
        self.idle = value

    def set_reading(self, value):
        self.reading = value

    """
    Sets sweep_start, the frequency to start sweep.
    """       
    def set_sweep_start(self, value):
        self.sweep_start = int(value)

    """
    Sets sweep_stop, the frequency to end sweep.
    """
    def set_sweep_stop(self, value):
        self.sweep_stop = int(value)

    """
    Sets stepValue. stepValue is the step frequency at which we increment during sweep. We DO NOT INPUT THIS NUMBER TO RP
    """
    def set_step_value(self):
        self.stepValue = np.round(1000 * (self.sweep_stop - self.sweep_start) / self.sweep_size, 3)

    """
    Sets sweep_size. sweep_size is the total number of frequency points in the scan
    """
    def set_sweep_size(self, value):
        self.sweep_size = int(value)
        # if self.stepValue == 0: return
        # self.sweep_size = int(1000 * (self.sweep_stop - self.sweep_start) / self.stepValue)

        # if self.stepValue == 0: return
        # self.sweep_size = int(1000 * (self.sweep_stop - self.sweep_start) / self.stepValue)
        # if self.sweep_size > 32766:
        #     self.sweep_size = 32766
        #     self.stepValue = int((1000 * (self.sweep_stop - self.sweep_start)) / 32766 + 0.5)
        #     if self.stepValue <= 1: self.stepValue = 1
        # print(self.sweep_size)

    
    def set_addressValue(self, value):
        self.addressValue = value.strip()
        print(self.addressValue)

    def set_reim(self, value):
        self.reim = value

    def set_connection_status(self, value):
        self.connection_status = value

    """
    Get methods
    """
    def get_stepValue(self):
        return str(self.stepValue)
    
    def get_rate(self):
        return self.rate
    
    def get_voltage(self):
        return self.voltage
    
    def get_reading(self):
        return self.reading
    
    def get_elapsedTime(self):
        return self.elapsedTime

    def get_sweep_size(self):
        return self.sweep_size

    def get_addressValue(self):
        return self.addressValue

    def get_sweep_start(self):
        return self.sweep_start

    def get_sweep_stop(self):
        return self.sweep_stop

    def get_socketError(self):
        return self.socketError

    def get_reim(self):
        return self.reim

    def get_socket(self):
        return self.socket

    def get_connect_status(self):
        return self.connect_status
    
    def get_count(self):
        return self.count
