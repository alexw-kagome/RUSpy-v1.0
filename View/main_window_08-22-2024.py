# Control program for the Red Pitaya RUS experiment
# Copyright (C) 2024  Alexander Won
# Based on the data acquisition code from Albert Migliori and Red Pitaya vector network analyzer from Pavel Demin
#
# Ref: Reviews of Scientific Instruments Vol.90 Issue 12 Dec 2019 pgs. 121401 ff.
# Copyright (C) 2022  Albert Migliori
# Based heavily on the Control program for the Red Pitaya vector network analyzer
# Copyright (C) 2021  Pavel Demin

import sys
from Controller.rus import rus
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import time
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

from lorentzian_fitting import FitResonances

from PyQt5.QtCore import QSettings, QTimer, Qt, QThread, pyqtSignal, QObject, QMutex, QMutexLocker
from PyQt5.QtNetwork import QTcpSocket
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QSizePolicy, QErrorMessage, QTableWidgetItem
from pathlib import Path
from PyQt5 import uic

# from PyQt5.QtCore import Null, QRegExp, QTimer, QSettings, QDir, Qt
from PyQt5.QtGui import QRegExpValidator, QPalette, QColor, QBitmap, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import (QApplication, QDoubleSpinBox, QMainWindow, QMessageBox, QDialog, QFileDialog, QPushButton,
                             QLabel, QSpinBox)
# from PyQt5.QtNetwork import QAbstractSocket, QTcpSocket
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplashScreen, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

'''
Some weird thing that apparently fixes pyqtgraph scaling with 1080p
'''
import platform
import ctypes
if platform.system()=='Windows' and int(platform.release()) >= 8:   
    ctypes.windll.shcore.SetProcessDpiAwareness(True)

'''
Suppressing DeprecationWarning for self.pbarTimer.start(self.delay)
'''
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
        
'''
Some weird thing that sets the taskbar icon to be the image I set.
Details - https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105%3E
'''
import ctypes
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        base_dir = Path(__file__).parent
        ui_file = base_dir / 'main.ui'
        logo_file = os.path.dirname(os.path.realpath(__file__))
        uic.loadUi(ui_file, self)
        
        self.setWindowIcon(QIcon(logo_file + os.path.sep + 'logo.png'))
        self.setWindowTitle('RUS')
        self.showMaximized()

        self.progressBar.setValue(0)
        self.pbarVal = 0
        self.pbarStep = 0
        self.sweepsize = 0
        self.rate = 0
        self.delay = 0
        self.const = 0
        self.num = 0
        self.count = 0
        self.lpcutoff = 3e-5
        self.hpcutoff = 1e-3

        self.exp = rus()

        self.exp.set_addressValue('10.84.241.73')
        self.addressValue.setText(str(self.exp.get_addressValue()))
        self.startFreqValue.setValue(1000)
        self.stopFreqValue.setValue(2500)
        self.pointValue.setText(str(32766))
        self.voltageValue.setValue(2)
        self.lpFilterValue.setText(str(self.lpcutoff))
        self.hpFilterValue.setText(str(self.hpcutoff))
        self.update_point_value()
        self.update_startfreq_value()
        self.update_stopfreq_value()
        self.update_volt_value()
        print(self.exp.get_sweep_size())
        self.rateValue.addItems(['1500', '500', '150', '50', '15', '5', '1.5', '0.5'])
        self.set_enabled(False)

        # Pyqtgraph widget
        self.layout = QVBoxLayout(self.plot_widget)
        self.pw = pg.PlotWidget(name='RUS Data')
        self.layout.addWidget(self.pw)
        self.p1 = self.pw.plot()
        self.p1.setPen((200,200,100))
        self.current_plot = 2
        self.current_filter = 1
        self.current_scale = 'normal'

        # connect signals from widgets
        self.plotButton.clicked.connect(self.switch_plot)
        self.connectButton.clicked.connect(self.connect_button_clicked)
        self.saveDatButton.clicked.connect(self.save)
        self.startButton.clicked.connect(self.start_sweep)
        self.stopButton.clicked.connect(self.stop_sweep)
        self.addressValue.textChanged.connect(self.exp.set_addressValue)
        self.filterButton.clicked.connect(self.switch_filter)
        # self.logButton.clicked.connect(self.switch_scale)
        self.readDatButton.clicked.connect(self.read_data)

        self.rateValue.currentIndexChanged.connect(self.exp.set_rate)

        self.voltageValue.valueChanged.connect(self.update_volt_value)
        self.pointValue.textChanged.connect(self.update_point_value)
        self.startFreqValue.valueChanged.connect(self.update_startfreq_value)
        self.stopFreqValue.valueChanged.connect(self.update_stopfreq_value)

        # create timers
        self.connectTimer = QTimer()
        self.connectTimer.timeout.connect(self.check_connection)

        self.sweepTimer = QTimer()
        self.sweepTimer.timeout.connect(self.check_sweep)

        self.pbarTimer = QTimer()
        self.pbarTimer.setSingleShot(True)
        self.pbarTimer.timeout.connect(self.update_pbar)

    def openbinfile(self, filename):
        dat = open(filename, 'r')
        data = np.fromfile(dat, dtype=np.dtype('>f8'))  # the raw data are binary files, so this imports them
        split_index = int((len(data) - 1) / 3)
        frequency = data[1: split_index + 1]
        x = data[split_index + 1: 2 * split_index + 1]
        y = data[2 * split_index + 1: 3 * split_index + 1]
        return frequency, x, y
    
    
    def read_data(self, filename):
        initial_directory = r'D:\Alexander'
        options = QFileDialog.Options()
        filenames, _ = QFileDialog.getOpenFileNames(
            self, 
            'Open File', 
            initial_directory,
            'All Files (*);;Data Files (*.dat);;Binary Files (*.bin)', 
            options=options
        )
        print(filenames)
        filename = str(filenames[0])
        if os.path.splitext(filename)[1] == '.dat':
            dat = np.loadtxt(filename)
            self.freq_raw, self.real_raw, self.imag_raw = dat[:, 0], dat[:, 1], dat[:, 2]
        elif os.path.splitext(filename)[1] == '.bin':
            self.freq_raw, self.real_raw, self.imag_raw = self.openbinfile(filename)
        else:
            print(os.path.splitext(filename)[1])

        self.realData, self.imagData, self.freq = self.interpolate_sweep(self.real_raw, self.imag_raw, self.freq_raw)
        self.plot_data()

    def save(self):
        path = os.getcwd()
        name = os.path.join(path, f'{self.exp.get_rate()}_rate_{self.exp.get_stepValue()}_step_{self.exp.get_sweep_size()}_sweepsize.dat')
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Data', name, 'Data (*.dat)')
        f = self.freq
        d = self.data
        if filename:
            if (d.real[0]) and (d.imag[0]) != 0.0:
                fh = open(filename, 'w')
                fh.write('#' + str(self.exp.get_rate()))
                fh.write(' rate (or translate to points/s)' +'\n')
                fh.write('#' + str(self.exp.get_stepValue()))
                fh.write(' step size (kHz)' +'\n')
                fh.write('#' + str(self.exp.get_sweep_size()))
                fh.write(' sweep size' +'\n')
                fh.write('#' + str(self.exp.get_voltage()))
                fh.write(' V' +'\n')
                fh.write('#' + str(self.exp.get_elapsedTime()))
                fh.write(' s' +'\n')
                fh.write('#  frequency        real         imag\n')
                for i in range(len(f)):
                    fh.write('%12.5f  %12.10f  %12.10f\n' % (f[i], d.real[i], d.imag[i]))
                fh.close()
    '''    
    def save(self):
        path = os.getcwd()
        name = os.path.join(path, f'{self.exp.get_rate()}_rate_{self.exp.get_stepValue()}_step_{self.exp.get_sweep_size()}_sweepsize.dat')
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Data', name, 'Data (*.dat)')
        reim = self.exp.get_reim()
        f = reim.freq
        d = reim.data
        if filename:
            if (d.real[0]) and (d.imag[0]) != 0.0:
                fh = open(filename, 'w')
                fh.write('#' + str(self.exp.get_rate()))
                fh.write(' rate (or translate to points/s)' +'\n')
                fh.write('#' + str(self.exp.get_stepValue()))
                fh.write(' step size (Hz)' +'\n')
                fh.write('#' + str(self.exp.get_sweep_size()))
                fh.write(' sweep size' +'\n')
                fh.write('#' + str(self.exp.get_voltage()))
                fh.write(' V' +'\n')
                fh.write('#' + str(self.exp.get_elapsedTime()))
                fh.write(' s' +'\n')
                fh.write('#  frequency        real         imag\n')
                for i in range(len(f)):
                    fh.write('%12.5f  %12.10f  %12.10f\n' % (f[i], d.real[i], d.imag[i]))
                fh.close()'''

    def plot_data(self):
        self.pw.clear()
        n = 30
        if self.current_filter == 1:
            data1, data2 =  self.realData[n:],  self.imagData[n:]
            data1 = data1 - np.mean(data1)
            data2 = data2 - np.mean(data2)
            data3 = np.sqrt(data1 ** 2 + data2 ** 2)
            frequency = self.freq[n:]
        else:
            data1, data2 = self.bandpass(self.realData, self.imagData, self.freq, self.lpcutoff, self.hpcutoff)
            data1, data2 = data1[n:], data2[n:]
            data1 = data1 - np.mean(data1)
            data2 = data2 - np.mean(data2)
            data3 = np.sqrt(data1 ** 2 + data2 ** 2)            
            frequency = self.freq[n:]

        # if self.current_scale == 'log':
        #     self.pw.setLogMode(y=True)
        #     self.pw.enableAutoRange(axis = 'x')
        #     self.pw.enableAutoRange(axis = 'y')
        # else:
        #     self.pw.setLogMode(y=False)
        #     self.pw.enableAutoRange(axis = 'x')
        #     self.pw.enableAutoRange(axis = 'y')

        styles = {'color': 'white', 'font-size': '30px'}

        if self.current_plot == 1:
            pen = pg.mkPen(color=(255, 0, 0), width=3, style=QtCore.Qt.SolidLine)
            self.pw.setLabel('left', 'Amplitude', units='V', **styles)
            self.pw.setLabel('bottom', 'Frequency', units='Hz', **styles)
            self.pw.setTitle('Magnitude', color='w', size='20pt')
            self.pw.plot(y=data3, x=frequency, pen=pen)
            plot_item = self.pw.getPlotItem()
            plot_item.showGrid(True, True, alpha = 0.5)
            plot_item.getAxis('bottom').setPen(pg.mkPen(color = 'w', width = 2))
            plot_item.getAxis('left').setPen(pg.mkPen(color = 'w', width = 2))
        else:
            self.pw.setLabel('left', 'Amplitude', units='V', **styles)
            self.pw.setLabel('bottom', 'Frequency', units='Hz', **styles)
            self.pw.setTitle('Real / Img', color='w', size='20pt')
            pen = pg.mkPen(color=(255, 0, 0), width=3, style=QtCore.Qt.SolidLine)
            self.pw.plot(y=data1, x=frequency, pen=pen)
            pen = pg.mkPen(color=(0, 0, 255), width=3, style=QtCore.Qt.SolidLine)
            self.pw.plot(y=data2, x=frequency, pen=pen)
            plot_item = self.pw.getPlotItem()
            plot_item.showGrid(True, True, alpha = 0.5)
            plot_item.getAxis('bottom').setPen(pg.mkPen(color = 'w', width = 2))
            plot_item.getAxis('left').setPen(pg.mkPen(color = 'w', width = 2))

    def interpolate_sweep(self, real, img, freq, delta_f=None):
        f_min, f_max = np.min(freq), np.max(freq)
        # df = abs(f_max - f_min)/np.shape(sweep.freq)[0]
        df = np.median(freq[1:]-freq[:-1])
        print(df)
        if not delta_f is None: df = delta_f
        N = int((f_max-f_min)/df)
        print(N)
        interp_f = np.linspace(f_min, f_max, N)
        Xint, Yint = interp1d(freq, real)(interp_f), interp1d(freq, img)(interp_f)
        return Xint, Yint, interp_f
    
    def bandpass(self, real, img, freq, nyq_low=0.00001, nyq_high=1):
        f, X, Y = freq, real, img
        df = abs(np.max(f)-np.min(f))/np.shape(f)[0]

        # high pass filter
        nyq = 2*nyq_low*df
        fb, fa = butter(3, nyq, btype= 'hp', analog= False)
        Xlp = filtfilt(fb, fa, X)
        Ylp = filtfilt(fb, fa, Y)

        # low pass filter after high pass filter
        nyq = 2*nyq_high*df
        if nyq>=1:
            Xbp, Ybp = Xlp, Ylp
        else:
            fb, fa = butter(3, nyq, btype= 'lp', analog= False)
            Xbp = filtfilt(fb, fa, Xlp)
            Ybp = filtfilt(fb, fa, Ylp)
        
        return Xbp, Ybp
    
    # def switch_scale(self):
    #     if self.current_scale == 'log':
    #         self.current_scale = 'normal'
    #         self.logButton.setText('Normal')
    #     else: 
    #         self.current_scale = 'log'
    #         self.logButton.setText('Logarithmic')
    #     self.plot_data()

    def switch_plot(self):
        if self.current_plot == 1:
            self.current_plot = 2
            self.plotButton.setText('See Magnitude')
        else:
            self.current_plot = 1
            self.plotButton.setText('See Real/Img')
        self.plot_data()

    def switch_filter(self):
        self.lpcutoff = float(self.lpFilterValue.text())
        self.hpcutoff = float(self.hpFilterValue.text())
        if self.current_filter == 1:
            self.current_filter = 2
            self.filterButton.setText('See Raw Data')
        else:
            self.current_filter = 1
            self.filterButton.setText('Remove Background')
        self.plot_data()
    
    def timeout(self):
        self.display_error('timeout')

    def display_error(self, socketError):
        self.startTimer.stop()
        if socketError == 'timeout':
            QMessageBox.information(self, 'rus', 'Error: connection timeout.')
        else:
            QMessageBox.information(self, 'rus', 'Error: %s.' % self.socket.errorString())
        self.stop()

    def start_sweep(self):
        start_time = time.time()
        isReady = self.sweep_condition()
        if isReady == True:
            self.progressBar.setValue(0)
            self.exp.set_rate(self.rateValue.currentIndex())

            self.update_point_value()
            self.update_startfreq_value()
            self.update_stopfreq_value()
            self.update_volt_value()

            self.set_enabled(False)

            self.sweepsize = self.exp.get_sweep_size()
            self.rate = self.exp.get_rate()

            if self.rate == 30:
                self.const = 1200
            elif self.rate == 100:
                self.const = 450
            elif self.rate == 300:
                self.const = 160
            elif self.rate == 1000:
                self.const = 50
            elif self.rate == 3000:
                self.const = 16
            elif self.rate == 10000:
                self.const = 5
            elif self.rate == 30000:
                self.const = 1.6
            elif self.rate == 100000:
                self.const = 0.5

            self.num = 1000
            self.delay = 1000 * self.sweepsize / self.const / self.num
            # print(self.delay)
            self.progressBar.setMaximum(self.num)
            self.pbarStep = 1
            self.pbarVal = 0
            self.count = 0

            self.sweepTimer.start(100)
            print(self.delay)
            self.pbarTimer.start(int(self.delay + 0.5))
            print('GUI')
            print(time.time() - start_time)
            self.exp.sweep()
        else:
            QMessageBox.information(self, 'rus', 'Error: Sweep size exceeds 32766 or incorrect sweep range.')
    
    def sweep_condition(self):
        if self.exp.get_sweep_start() < self.exp.get_sweep_stop():
            if self.exp.get_sweep_size() <= 32766:
                if self.exp.get_voltage() <= 2:
                    return True
        return False
    
    def print_params(self):
        print(self.exp.get_stepValue())
        print(self.exp.get_sweep_start())
        print(self.exp.get_sweep_stop())
        print(self.exp.get_sweep_size())
        print(self.exp.get_voltage())

    def update_pbar(self):
        if self.num > self.count:
            # print('update pbar')
            self.pbarVal = self.pbarVal + self.pbarStep
            self.progressBar.setValue(self.pbarVal)
            self.count = self.count + 1
            self.pbarTimer.start(int(self.delay + 0.5))


    def check_sweep(self):
        if self.exp.get_reading() == False:
            if self.exp.get_elapsedTime() != 0:
                self.sweepTimer.stop()
                self.pbarTimer.stop()
                self.progressBar.setValue(self.num)

                self.freq_raw = self.exp.get_reim().freq * 1000
                self.data_raw = self.exp.get_reim().data
                self.real_raw = np.real(self.data_raw)
                self.imag_raw = np.imag(self.data_raw)
                self.realData, self.imagData, self.freq = self.interpolate_sweep(self.real_raw, self.imag_raw, self.freq_raw)
                self.plot_data()
                self.set_enabled(True)

    def stop_sweep(self): 
        self.exp.cancel()
        self.sweepTimer.stop()
        self.pbarTimer.stop()
        self.set_enabled(True)

    def disconnect(self):
        self.exp.cancel()
        self.exp.stop()
        self.sweepTimer.stop()
        self.pbarTimer.stop()
        self.set_enabled(True)

    def connect_button_clicked(self):
        if self.connectButton.text() == 'Connect':
            self.disconnect()
            self.set_enabled(False)
            self.exp.start(self.exp.get_addressValue())
            self.connectTimer.start(100)
        else:
            self.disconnect()
            self.exp.set_connection_status('')
            self.set_enabled(False)
            self.connectButton.setText('Connect')

    def check_connection(self):
        # self.print_params()
        if self.exp.get_connect_status() != '':
            self.connectTimer.stop()
            if self.exp.get_connect_status() == 'connected':
                self.connectButton.setText('Disconnect')
                self.connectButton.setEnabled(True)
                self.set_enabled(True)
                print('CONNECTED')
                return
            if self.exp.get_socketError() == 'timeout':
                QMessageBox.information(self, 'rus', 'Error: connection timeout.')
                self.exp.set_connection_status('')
            else:
                QMessageBox.information(self, 'rus', 'Error: %s.' % self.exp.get_connect_status())
                self.exp.set_connection_status('')
    
    def set_enabled(self, enabled):
        widgets = [self.rateValue, self.voltageValue, self.pointValue, self.startFreqValue,
                   self.stopFreqValue, self.startButton]
        for entry in widgets:
            entry.setEnabled(enabled)

    def update_volt_value(self):
        self.exp.set_volt1(self.voltageValue.value())
        self.voltageValue.setValue(self.exp.get_voltage())

    def update_point_value(self):
        self.exp.set_sweep_size(int(self.pointValue.text()))
        self.exp.set_step_value()
        self.stepValue.setText(self.exp.get_stepValue())

    def update_startfreq_value(self):
        self.exp.set_sweep_start(self.startFreqValue.value())
        self.startFreqValue.setValue(self.exp.get_sweep_start())
        self.exp.set_step_value()
        self.stepValue.setText(self.exp.get_stepValue())

    def update_stopfreq_value(self):
        self.exp.set_sweep_stop(self.stopFreqValue.value())
        self.stopFreqValue.setValue(self.exp.get_sweep_stop())
        self.exp.set_step_value()
        self.stepValue.setText(self.exp.get_stepValue())

class resonanceDetector(QWidget):
    variableChanged = pyqtSignal()
    def __init__(self):
        super().__init__()
        base_dir = Path(__file__).parent
        ui_file = base_dir / 'resonance_detector.ui'
        uic.loadUi(ui_file, self)

        logo_file = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(logo_file + os.path.sep + 'logo.png'))
        
        self.fitTimer = QTimer()
        self.fitTimer.setSingleShot(True)
        self.fitTimer.timeout.connect(self.fit_timeout)
        
        self.variableChanged.connect(self.vline_changed)

        self.setWindowTitle('Resonance Detector')
        self.showMaximized()

        self.filepath = ''
        self.real_raw, self.imag_raw, self.freq_raw = [], [], []
        self.data1, self.data2, self.data3, self.freq = [], [], [], []

        # Pyqtgraph widget
        self.layout1 = QVBoxLayout(self.widget1)
        self.pw1 = pg.PlotWidget(name='')
        self.layout1.addWidget(self.pw1)
        self.p1 = self.pw1.plot()
        self.p1.setPen((200,200,100))

        self.layout2 = QVBoxLayout(self.widget2)
        self.pw2 = pg.PlotWidget(name='Amplitude')
        self.layout2.addWidget(self.pw2)
        self.p2 = self.pw2.plot()
        self.p2.setPen((200,200,100))
        self.plot_scale = 2
        self.resultTable.setRowCount(9)
        self.resultTable.setColumnCount(1)        
        self.resultTable.setHorizontalHeaderLabels(['Value'])
        self.resultTable.setVerticalHeaderLabels(['f0', 'Gamma', 'Q', 'A', 'Phi',
                                                'X background intercept',
                                                'X background slope', 
                                                'Y background intercept',
                                                'Y background slope'])
        
        for i in range(9):
             self.resultTable.setItem(i, 0, QTableWidgetItem(0))
        
        self.saveTable.setRowCount(200)
        self.saveTable.setColumnCount(9)
        self.saveTable.setHorizontalHeaderLabels(['f0', 'Gamma', 'Q', 'A', 'Phi',
                                                'X background intercept',
                                                'X background slope', 
                                                'Y background intercept',
                                                'Y background slope'])
        
        self.vline1 = pg.InfiniteLine(pos = 10000, angle=90, movable=True, pen=pg.mkPen(color = (40, 252, 3)))
        self.vline2 = pg.InfiniteLine(pos = 20000, angle=90, movable=True, pen=pg.mkPen(color = (40, 252, 3)))
        self.last_position = {self.vline1: self.vline1.value(), self.vline2: self.vline2.value()}
        self.syncing = False

        self.current_row = 0
        self.highlight_row()
        
        self.lpcutoff = 3e-5
        self.hpcutoff = 1e-3
        self.lpcutofftemp = 3e-5
        self.hpcutofftemp = 1e-3

        self.lpFilterValue.setText(str(self.lpcutoff))
        self.hpFilterValue.setText(str(self.hpcutoff))
        self.lpSlider.setRange(-1000, 1000)
        self.lpSlider.setValue(0)
        self.hpSlider.setRange(-1000, 1000)
        self.hpSlider.setValue(0)

        self.vline1.sigPositionChanged.connect(self.sync_lines)
        self.vline2.sigPositionChanged.connect(self.sync_lines)
        self.browseButton.clicked.connect(self.browsefiles)
        self.lpFilterValue.editingFinished.connect(self.update_cutoff)
        self.hpFilterValue.editingFinished.connect(self.update_cutoff)
        self.lpSlider.valueChanged.connect(self.update_lpslider)
        self.hpSlider.valueChanged.connect(self.update_hpslider)
        self.filterResetButton.clicked.connect(self.reset_filter)
        self.rawButton.clicked.connect(self.raw_button)

    def browsefiles(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', r'D:\Alexander')
        self.filepath = fname[0]
        self.fileAddress.setText(self.filepath)
        self.readfile()
        self.plot_data()
        self.plot_mag()

    def openbinfile(self, filename):
        dat = open(filename, 'r')
        data = np.fromfile(dat, dtype=np.dtype('>f8'))  # the raw data are binary files, so this imports them
        split_index = int((len(data) - 1) / 3)
        frequency = data[1: split_index + 1]
        x = data[split_index + 1: 2 * split_index + 1]
        y = data[2 * split_index + 1: 3 * split_index + 1]
        return frequency * 1000, x, y
    
    def readfile(self):
        if os.path.splitext(self.filepath)[1] == '.dat':
            dat = np.loadtxt(self.filepath)
            self.freq_raw, self.real_raw, self.imag_raw = dat[:, 0] * 1000, dat[:, 1], dat[:, 2]
        elif os.path.splitext(self.filepath)[1] == '.bin':
            self.freq_raw, self.real_raw, self.imag_raw = self.openbinfile(self.filepath)

        self.data1 = self.real_raw[10:]
        self.data2 = self.imag_raw[10:]
        self.data3 = np.sqrt(self.data1 ** 2 + self.data2 ** 2)
        self.freq = self.freq_raw[10:]

    def bandpass(self, freq, data, nyq_low=0.00001, nyq_high=1):
        f, X = freq, data
        df = abs(np.max(f)-np.min(f))/np.shape(f)[0]

        # high pass filter
        nyq = 2*nyq_low*df
        fb, fa = butter(3, nyq, btype= 'hp', analog= False)
        Xlp = filtfilt(fb, fa, X)

        # low pass filter after high pass filter
        nyq = 2*nyq_high*df
        if nyq>=1:
            Xbp = Xlp
        else:
            fb, fa = butter(3, nyq, btype= 'lp', analog= False)
            Xbp = filtfilt(fb, fa, Xlp)
        
        return Xbp
    
    def plot_data(self):
        mask = (self.freq >= self.last_position[self.vline1]) & (self.freq <= self.last_position[self.vline2])
        data1 = self.data1[mask]
        data2 = self.data2[mask]
        freq = self.freq[mask]       

        if len(freq) != 0:
            self.pw1.clear()
            
            styles = {'color': 'white', 'font-size': '28px'}
            self.pw1.setLabel('left', 'Amplitude', units='V', **styles)
            self.pw1.setLabel('bottom', 'Frequency', units='Hz', **styles)
            self.pw1.setTitle('Real / Img', color='w', size='18pt')

            pen_red = pg.mkPen(color=(255, 0, 0), width=0.3, style=QtCore.Qt.SolidLine)
            pen_blue = pg.mkPen(color=(0, 0, 255), width=0.3, style=QtCore.Qt.SolidLine)
            self.pw1.plot(x=freq, y=data1 - np.average(data1), pen=None, symbol='o', symbolBrush='r', symbolPen=pen_red)
            self.pw1.plot(x=freq, y=data2 - np.average(data2), pen=None, symbol='o', symbolBrush='b', symbolPen=pen_blue)


            plot_item = self.pw1.getPlotItem()
            plot_item.showGrid(True, True, alpha = 0.5)
            plot_item.getAxis('bottom').setPen(pg.mkPen(color = 'w', width = 2))
            plot_item.getAxis('left').setPen(pg.mkPen(color = 'w', width = 2))
            self.pw1.enableAutoRange(axis = 'x')
            self.pw1.enableAutoRange(axis = 'y')
    
    def table(self, row = True):
        data = [self.w0, self.gamma, np.pi * self.w0 / self.gamma, self.A, self.phi, self.c0, self.m0, self.c1, self.m1]
        if row:
            for index, value in enumerate(data):
                self.resultTable.setItem(index, 0, QTableWidgetItem(str(np.round(value, 5))))
        else:
            for index, value in enumerate(data):
                self.saveTable.setItem(self.current_row, index, QTableWidgetItem(str(value), 5))
            self.move_row(1)      
            self.highlight_row()   

    def remove_row(self):
        if self.saveTable.rowCount() > 0 and self.current_row < self.saveTable.rowCount():
            self.saveTable.removeRow(self.current_row)
            self.highlight_row()

    def move_row(self, direction):
        if direction == -1 and self.current_row > 0:
            self.current_row -= 1
        elif direction == 1 and self.current_row < self.saveTable.rowCount() - 1:
            self.current_row += 1
        self.highlight_row()

    def highlight_row(self):
        self.saveTable.clearSelection()  
        if 0 <= self.current_row < self.saveTable.rowCount():
            self.saveTable.setCurrentCell(self.current_row, 0)  

    def save_table(self):
        initial_directory = r'D:\Alexander'
        default_filename = 'resonances.dat'
        headers = ['f0', 'Gamma', 'Q', 'A', 'Phi',
            'X background intercept',
            'X background slope', 
            'Y background intercept',
            'Y background slope'
        ]

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Table", 
            f"{initial_directory}/{default_filename}", 
            "Data Files (*.dat);;All Files (*)", 
            options=options
        )

        if file_name:
            with open(file_name, 'w') as file:
                header_line = '# ' + '\t'.join(headers) + '\n'
                file.write(header_line)
                for row in range(self.saveTable.rowCount()):
                    line = '\t'.join(
                        self.saveTable.item(row, col).text() if self.saveTable.item(row, col) else ''
                        for col in range(self.saveTable.columnCount())
                    )
                    file.write(line + '\n')

    def plot_fit(self):
        mask = (self.freq >= self.last_position[self.vline1]) & (self.freq <= self.last_position[self.vline2])
        data1 = self.data1[mask]
        data2 = self.data2[mask]
        freq = self.freq[mask]

        if len(freq) != 0:
            fit = FitResonances(freq, data1, data2)
            # self.w0, self.gamma, A, phi = fit.arkady()
            self.w0, self.gamma, self.A, self.phi, self.c0, self.c1, self.m0, self.m1 = fit.single_lorentzian_fitting()
            x_vals, data1_fit, data2_fit = fit.plot_lorentzian(self.w0, self.gamma, self.A, self.phi, self.c0, self.c1, self.m0, self.m1)
            data1 = data1 - self.c0 - self.m0 * freq
            data2 = data2 - self.c1 - self.m1 * freq

            self.table(True)

            self.pw1.clear()
            self.pw1.enableAutoRange(axis = 'x')
            self.pw1.enableAutoRange(axis = 'y')

            styles = {'color': 'white', 'font-size': '28px'}
            self.pw1.setLabel('left', 'Amplitude', units='V', **styles)
            self.pw1.setLabel('bottom', 'Frequency', units='Hz', **styles)
            self.pw1.setTitle('Real / Img', color='w', size='18pt')

            # pen = pg.mkPen(color=(255, 0, 0), width=3, style=QtCore.Qt.SolidLine)
            # self.pw1.plot(y=data2, x=freq, pen=pen)
            # pen = pg.mkPen(color=(0, 0, 255), width=3, style=QtCore.Qt.SolidLine)
            # self.pw1.plot(y=data1, x=freq, pen=pen)

            pen_red = pg.mkPen(color=(255, 0, 0), width=0.3, style=QtCore.Qt.SolidLine)
            pen_blue = pg.mkPen(color=(0, 0, 255), width=0.3, style=QtCore.Qt.SolidLine)
            self.pw1.plot(x=freq, y=data1, pen=None, symbol='o', symbolBrush='r', symbolPen=pen_red)
            self.pw1.plot(x=freq, y=data2, pen=None, symbol='o', symbolBrush='b', symbolPen=pen_blue)

            pen = pg.mkPen(color=(255, 255, 255), width=2, style=QtCore.Qt.SolidLine)
            self.pw1.plot(y=data1_fit, x=x_vals, pen=pen)
            self.pw1.plot(y=data2_fit, x=x_vals, pen=pen)

            plot_item = self.pw1.getPlotItem()
            plot_item.showGrid(True, True, alpha = 0.5)
            plot_item.getAxis('bottom').setPen(pg.mkPen(color = 'w', width = 2))
            plot_item.getAxis('left').setPen(pg.mkPen(color = 'w', width = 2))

    def plot_mag(self, raw = False):
        x_min, x_max = self.pw2.getViewBox().viewRange()[0]
        self.df = (x_max - x_min) / 500
        if raw:
            data3 = self.data3
        else:
            data3 = self.bandpass(self.freq, self.data3, self.lpcutoff, self.hpcutoff)

        self.pw2.clear()
        self.pw2.enableAutoRange(axis = 'x')
        self.pw2.enableAutoRange(axis = 'y')

        self.vline1.setValue(self.freq[int(len(self.freq) / 3)])
        self.vline2.setValue(self.freq[int(len(self.freq) / 2)])
        self.last_position = {self.vline1: self.vline1.value(), self.vline2: self.vline2.value()}
        self.pw2.addItem(self.vline1)
        self.pw2.addItem(self.vline2)

        styles = {'color': 'white', 'font-size': '28px'}
        pen = pg.mkPen(color=(255, 0, 0), width=1, style=QtCore.Qt.SolidLine)
        self.pw2.setLabel('left', 'Amplitude', units='V', **styles)
        self.pw2.setLabel('bottom', 'Frequency', units='Hz', **styles)
        self.pw2.setTitle('Magnitude', color='w', size='18pt')
        self.pw2.plot(y = data3, x = self.freq, pen=pen)
        plot_item = self.pw2.getPlotItem()
        plot_item.showGrid(True, True, alpha = 0.5)
        plot_item.getAxis('bottom').setPen(pg.mkPen(color = 'w', width = 2))
        plot_item.getAxis('left').setPen(pg.mkPen(color = 'w', width = 2))
    
    def raw_button(self):
        self.plot_mag(True)

    def update_cutoff(self):
        self.lpcutoff = float(self.lpFilterValue.text())
        self.lpcutofftemp = float(self.lpFilterValue.text())
        self.hpcutoff = float(self.hpFilterValue.text())
        self.hpcutofftemp = float(self.hpFilterValue.text())
        self.plot_mag()

    def update_lpslider(self):
        self.lpcutoff = self.lpcutofftemp * 10 ** (self.lpSlider.value() / 1000)
        self.lpFilterValue.setText(str(self.lpcutoff))
        self.plot_mag()

    def update_hpslider(self):
        self.hpcutoff = self.hpcutofftemp * 10 ** (self.hpSlider.value() / 1000)
        self.hpFilterValue.setText(str(self.hpcutoff))
        self.plot_mag()
    
    def reset_filter(self):
        self.lpcutoff = 3e-5
        self.hpcutoff = 1e-3
        self.lpcutofftemp = 3e-5
        self.hpcutofftemp = 1e-3
        self.lpFilterValue.setText(str(self.lpcutoff))
        self.hpFilterValue.setText(str(self.hpcutoff))
        self.lpSlider.setValue(0)
        self.hpSlider.setValue(0)
        self.plot_mag()

    def vline_changed(self):
        self.fitTimer.start(1000)  #ms
    
    def fit_timeout(self):
        self.plot_fit()

    def sync_lines(self):
        x_min, x_max = self.pw2.getViewBox().viewRange()[0]
        self.df = (x_max - x_min) / 500
        self.plot_data()

        if self.syncing:
            return

        self.syncing = True

        # Get the current positions of the vertical lines
        pos1 = self.vline1.value()
        pos2 = self.vline2.value()

        # Determine the last known positions
        last_pos1 = self.last_position[self.vline1]
        last_pos2 = self.last_position[self.vline2]

        if pos1 != last_pos1:
            # vline1 was moved
            displacement = pos1 - last_pos1
            self.vline2.sigPositionChanged.disconnect(self.sync_lines)  # Temporarily disconnect
            self.vline2.setValue(last_pos2 + displacement)  # Move the second line
            self.vline2.sigPositionChanged.connect(self.sync_lines)  # Reconnect
        else:
            # vline2 was moved
            displacement = pos2 - last_pos2
            self.vline1.sigPositionChanged.disconnect(self.sync_lines)  
            self.vline1.setValue(last_pos1 + displacement)  
            self.vline1.sigPositionChanged.connect(self.sync_lines) 

        # Update the last positions
        self.last_position[self.vline1] = self.vline1.value()
        self.last_position[self.vline2] = self.vline2.value()
        self.variableChanged.emit()

        self.syncing = False

    def keyPressEvent(self, event):
        self.syncing = True
        key = event.key()

        if key == Qt.Key_O:
            self.table(False)
        elif key == Qt.Key_R and event.modifiers() & Qt.ShiftModifier:
            self.remove_row()
        elif key == Qt.Key_PageUp:
            self.move_row(-1)
        elif key == Qt.Key_PageDown:
            self.move_row(1)
        elif key == Qt.Key_S and event.modifiers() & Qt.ShiftModifier:
            self.save_table()
        elif key == QtCore.Qt.Key_Up:
            self.adjust_range(self.df)
        elif key == QtCore.Qt.Key_Down:
            self.adjust_range(-self.df)
        elif key == QtCore.Qt.Key_Left:
            self.move_lines(-self.df)
        elif key == QtCore.Qt.Key_Right:
            self.move_lines(self.df)

        self.last_position[self.vline1] = self.vline1.value()
        self.last_position[self.vline2] = self.vline2.value()
        self.variableChanged.emit()

        self.syncing = False
        
    def adjust_range(self, delta):
        v1_pos = self.vline1.value()
        v2_pos = self.vline2.value()

        if v2_pos - v1_pos > - 2 * delta:  # Minimum range
            self.vline1.setValue(v1_pos - delta)
            self.vline2.setValue(v2_pos + delta)

    def move_lines(self, delta):
        self.vline1.setValue(self.vline1.value() + delta)
        self.vline2.setValue(self.vline2.value() + delta)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    w = resonanceDetector()
    w.show()
    arg = app.exec_()
    window.disconnect()
    print('disconnected')
    sys.exit(arg)


if __name__ == '__main__':
    main()
