# Control program for the Red Pitaya RUS experiment
# Copyright (C) 2024  Alexander Won
# Based on the data acquisition code from Albert Migliori and Red Pitaya vector network analyzer from Pavel Demin
# Ref: Reviews of Scientific Instruments Vol.90 Issue 12 Dec 2019 pgs. 121401 ff.

from asyncio import sleep
from operator import pos
from datetime import datetime
import sys
import struct
import warnings
from matplotlib import pyplot as plt
import os
from pathlib import Path

import time
import numpy as np
import matplotlib


matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter, FuncFormatter
from matplotlib.widgets import Cursor, MultiCursor
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

from PyQt5.uic import loadUiType
from PyQt5 import QtCore
from PyQt5.QtCore import QRegExp, QTimer, QSettings, QDir, Qt, QObject, QElapsedTimer
from PyQt5.QtGui import QRegExpValidator, QPalette, QColor, QBitmap, QPixmap
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QMainWindow, QMessageBox, QDialog, QFileDialog, QPushButton, \
    QLabel, QSpinBox
from PyQt5.QtNetwork import QAbstractSocket, QTcpSocket
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplashScreen

'''
Measurement Class to store data
'''
class Measurement:
    def __init__(self):
        self.freq = np.array([], dtype=float)
        self.data = np.array([], dtype=np.complex64)

class RedPitaya(QObject):

    MAX_SWEEP_SIZE = 32766

    # Signal to emit to main_window
    data_chunk_ready = QtCore.pyqtSignal(np.ndarray)
    sweep_finished = QtCore.pyqtSignal()
    connected_signal = QtCore.pyqtSignal(str)
    tracking_sweep_progress = QtCore.pyqtSignal()

    def __init__(self):
        super(RedPitaya, self).__init__()

        self.initialize()
        
        '''
        TCP socket to communicate with RP.
        '''
        self.socket = QTcpSocket(self)
        self.socket.readyRead.connect(self.read_data)
        self.socket.connected.connect(self.connected)
        self.socket.error.connect(self.error)

        '''
        startTimer: Detect timeout during connection
        sweepTimer: Measures time took for experiment
        '''
        self.startTimer = QTimer(self)
        self.startTimer.timeout.connect(self.timeout)

        self.sweepTimer = QElapsedTimer()

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
        self.long_sweep_active = False

        # buffer and offset for the incoming samples
        self.buffer = bytearray(16 * 32768)
        self.offset = 0
        self.data = np.frombuffer(self.buffer, np.complex64)

        # create measurements
        self.reim = Measurement()

        self.tracking_sweep_finished = True
    
    '''
    RP is connected.
    '''
    def connected(self):
        self.startTimer.stop()
        self.connect_status = "connected"
        self.idle = False
        self.connected_signal.emit(self.connect_status)
        print('Connected')
        self.set_corr(0)
        self.set_phase1(0)
        self.set_phase2(0)
        self.set_volt2(0)
        self.set_gpio(1)

    '''
    RP timed out; update status to timeout.
    '''
    def timeout(self):
        self.idle = True
        self.connect_status = "timeout"
        self.connected_signal.emit(self.connect_status)
        self.startTimer.stop()

    '''
    RP throws error during connection.
    '''
    def error(self):
        self.idle = True
        self.connect_status = self.socket.errorString()
        self.connected_signal.emit(self.connect_status)
        self.startTimer.stop()

    '''
    connect to RP.
    '''
    def start(self, address):
        self.addressValue = address.strip()
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
        self.tracking_sweep_finished = True
        self.socket.write(struct.pack("<I", 11 << 28))

    def start_tracking_sweep(self, sweeps):
        self.tracking_sweep_queue = list(sweeps)
        self.tracking_sweep_finished = False
        self._tracking_sweep_index = 0
        if not hasattr(self, '_tracking_sweep_progress_connected'):
            self.tracking_sweep_progress.connect(self._on_tracking_sweep_progress)
            self._tracking_sweep_progress_connected = True
        self._start_next_tracking_sweep()

    def _start_next_tracking_sweep(self):
        if not hasattr(self, 'tracking_sweep_queue') or self._tracking_sweep_index >= len(self.tracking_sweep_queue):
            self.tracking_sweep_finished = True
            self.sweep_finished.emit()
            return
        sweep = self.tracking_sweep_queue[self._tracking_sweep_index]
        self.limit = 16 * int(sweep[2])
        self.numsample = max(5, int(sweep[2] / 1e3))
        self.start_long_sweep(sweep[0], sweep[1], int(sweep[2]))

    def _on_tracking_sweep_progress(self):
        self._tracking_sweep_index += 1
        self._start_next_tracking_sweep()

    def start_long_sweep(self, sweep_start, sweep_stop, sweep_size):
        """
        Multiple subsweeps
        """
        self.long_sweep_active = True
        self.long_sweep_start = sweep_start
        self.long_sweep_stop = sweep_stop
        self.long_sweep_size = sweep_size
        self.long_sweep_data = []
        self.long_sweep_freq = []
        self.long_sweep_index = 0
        self._start_next_subsweep()

    def _start_next_subsweep(self):
        total_size = self.long_sweep_size
        max_size = self.MAX_SWEEP_SIZE
        idx = self.long_sweep_index
        sub_size = min(max_size, total_size - idx * max_size)
        
        if sub_size <= 0:
            # sweep finished
            self.long_sweep_active = False
            all_data = np.concatenate(self.long_sweep_data)
            all_freq = np.concatenate(self.long_sweep_freq)
            self.reim.freq = np.concatenate((self.reim.freq, all_freq))
            self.reim.data = np.concatenate((self.reim.data, all_data))
            self.tracking_sweep_progress.emit()

            if self.tracking_sweep_finished:
                self.sweep_finished.emit()
            return
        
        # sub-sweep parameters
        start = self.long_sweep_start + (self.long_sweep_stop - self.long_sweep_start) * (idx * max_size) / total_size
        stop = self.long_sweep_start + (self.long_sweep_stop - self.long_sweep_start) * ((idx * max_size + sub_size - 1) / (total_size - 1))
        self.sweep_start = start
        self.sweep_stop = stop
        self.sweep_size = sub_size
        self.set_sweep_size(sub_size)
        self.long_sweep_index += 1
        self.offset = 0
        self.count = 0
        self.reading = True
        self.socket.write(struct.pack("<I", 0 << 28 | int(self.sweep_start)))
        self.socket.write(struct.pack("<I", 1 << 28 | int(self.sweep_stop)))
        self.socket.write(struct.pack("<I", 2 << 28 | int(self.sweep_size)))
        self.socket.write(struct.pack("<I", 10 << 28))

    '''
    read data from RP.
    read_data is called whenever socket is readyRead.
    note: we only retreive data once experiment is over.
    '''
    def read_data(self):
        self.count = self.count + 1

        while self.socket.bytesAvailable() > 0:
            if not self.reading:
                self.socket.readAll()
                return
            size = self.socket.bytesAvailable()
            bytes_to_read = min(size, self.limit - self.offset)
            self.buffer[self.offset:self.offset + bytes_to_read] = self.socket.read(bytes_to_read)
            prev_offset = self.offset
            self.offset += bytes_to_read
            # numsample samples, 16 bytes per sample (complex64)
            # Emit signal for every 'sample' samples received
            while prev_offset // 16 < self.offset // 16:
                start_sample = (prev_offset // 16) // self.numsample * self.numsample
                end_sample = min(((start_sample + self.numsample), self.offset // 16, self.sweep_size))
                if end_sample - start_sample == self.numsample:
                    chunk = self.data[start_sample * 2:(start_sample + self.numsample) * 2]  # 2 channels per sample
                    self.data_chunk_ready.emit(chunk.copy())
                    prev_offset = (start_sample + self.numsample) * 16
                else:
                    break

            # If sweep is over
            if self.offset >= self.limit:
                adc1 = self.data[0::2]
                start = self.sweep_start
                stop = self.sweep_stop
                size = self.sweep_size
                freq = np.linspace(start, stop, size)
                data = adc1[0:size].copy()

                if self.long_sweep_active == True:
                    self.long_sweep_data.append(data)
                    self.long_sweep_freq.append(freq)
                    self.reading = False
                    # Start next sub-sweep
                    self._start_next_subsweep()
                else:
                    attr = getattr(self, "reim")
                    attr.freq = freq
                    attr.data = data
                    attr.unity = np.full(size, 1)
                    self.reading = False
                    self.elapsedTime = self.sweepTimer.elapsed()*1e3
                    self.sweep_finished.emit()
                
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
        self.start_time = time.time()
        if self.idle: return
        self.offset = 0
        self.count = 0
        self.reading = True
        print(f"start: {self.sweep_start} stop: {self.sweep_stop} # data: {self.sweep_size}")
        self.socket.write(struct.pack("<I", 0 << 28 | int(self.sweep_start)))
        self.socket.write(struct.pack("<I", 1 << 28 | int(self.sweep_stop)))
        self.socket.write(struct.pack("<I", 2 << 28 | int(self.sweep_size)))
        self.socket.write(struct.pack("<I", 10 << 28))
        self.sweepTimer.start()

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
        self.rate = [30, 100, 300, 1000, 3000, 10000, 30000, 100000][int(value)]
        self.socket.write(struct.pack("<I", 3 << 28 | int(self.rate)))
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
        self.stepValue = np.round((self.sweep_stop - self.sweep_start) / self.sweep_size, 3)

    """
    Sets sweep_size. sweep_size is the total number of frequency points in the scan
    """
    def set_sweep_size(self, value):
        self.sweep_size = int(value)
        # Total number of samples to acquire
        self.limit = 16 * self.sweep_size
        # number of samples to live emit (at least 5)
        self.numsample = max(5, int(self.sweep_size / 1e3))

    def set_addressValue(self, value):
        self.addressValue = value.strip()

    def set_reim(self, value):
        self.reim = value

    def set_connection_status(self, value):
        self.connection_status = value

    def reset_reim(self):
        self.reim = Measurement()
        
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
