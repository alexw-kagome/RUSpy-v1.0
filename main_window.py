# Control program for the Red Pitaya RUS experiment
# Copyright (C) 2024  Alexander Won
# Based on the data acquisition code from Albert Migliori and Red Pitaya vector network analyzer from Pavel Demin
# Ref: Reviews of Scientific Instruments Vol.90 Issue 12 Dec 2019 pgs. 121401 ff.

from red_pitaya import RedPitaya
from cryocon_22C import Cryocon22C


from device_connection_window import deviceConnection
from resonance_detector_window import resonanceDetector
from resonance_tracking_window import resonanceTracking

import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import time
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from lorentzian_fitting import FitResonances

import twisted
from twisted.internet.defer import inlineCallbacks, Deferred

from PyQt5.QtCore import QSettings, QTimer, Qt, QThread, pyqtSignal, QObject, QMutex, QMutexLocker
from PyQt5.QtNetwork import QTcpSocket
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QMessageBox, QDialog, QFileDialog, QWidget, QVBoxLayout
from PyQt5.QtGui import QIcon
from pathlib import Path
from PyQt5 import uic
import time
from datetime import datetime

'''
Some weird thing that apparently fixes pyqtgraph scaling with 1080p
'''
import platform
import ctypes
if platform.system()=='Windows' and int(platform.release()) >= 8:   
    ctypes.windll.shcore.SetProcessDpiAwareness(True)

'''
Suppressing DeprecationWarning
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

    def __init__(self, reactor):
        super(MainWindow, self).__init__()
        base_dir = Path(__file__).parent
        ui_file = base_dir / 'main.ui'
        logo_file = os.path.dirname(os.path.realpath(__file__))
        uic.loadUi(ui_file, self)
        
        self.reactor = reactor

        self.setWindowIcon(QIcon(logo_file + os.path.sep + 'logo.png'))
        self.setWindowTitle('RUS Sweeper')
        self.showMaximized()

        self.device_address = {
            'Red Pitaya': '10.84.241.73',
            'Cryocon': 'COM3'
        }

        self.device_status = {
            'Red Pitaya': None,
            'Cryocon': None
        }

        self.progressBar.setValue(0)
        self.rate = 0
        self.lpcutoff = 3e-5
        self.hpcutoff = 1e-3
        self.startfreq, self.stopfreq, self.sweepsize = 0, 0, 0
        self.temp, self.t0, self.t1 = 0, 0, 0

        self.rp = RedPitaya()
        self.cryo = Cryocon22C(self.reactor, port=self.device_address['Cryocon'])

        self.startFreqValue.setValue(1)
        self.stopFreqValue.setValue(2)
        self.pointValue.setText(str(50))
        self.voltageValue.setValue(2)
        self.lpFilterValue.setText(str(self.lpcutoff))
        self.hpFilterValue.setText(str(self.hpcutoff))
        self.update_point_value()
        self.update_startfreq_value()
        self.update_stopfreq_value()
        self.update_volt_value()
        self.rateValue.addItems(['1500', '500', '150', '50', '15', '5', '1.5', '0.5'])
        self.rateValue.setCurrentText('150')
        self.set_enabled(False)
        self.saveDirLine.setText('C:/Users/lab29/Downloads/test')
        self.save_path = r'C:/Users/lab29/Downloads/test'

        # Continuous sweep related
        self.current_sweep = 0

        # Resonance Tracking
        self.resonances_to_track = np.array([])
        self.tracking_window = 0
        self.dense_spacing = 0
        self.sparse_spacing = 0

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
        self.saveCurrentButton.clicked.connect(self.save_sweep)
        self.saveDirButton.clicked.connect(self.save_dir)
        self.startButton.clicked.connect(self.start_sweep)
        self.stopButton.clicked.connect(self.stop_sweep)
        self.filterButton.clicked.connect(self.switch_filter)
        self.readDatButton.clicked.connect(self.read_data)

        self.deviceConnectionButton.clicked.connect(self.open_device_connection)
        self.resonanceDetectorButton.clicked.connect(self.open_resonance_detector)
        self.resonanceTrackingButton.clicked.connect(self.open_resonance_tracking)

        self.rateValue.currentIndexChanged.connect(self.rp.set_rate)

        self.voltageValue.valueChanged.connect(self.update_volt_value)
        self.pointValue.textChanged.connect(self.update_point_value)
        self.startFreqValue.valueChanged.connect(self.update_startfreq_value)
        self.stopFreqValue.valueChanged.connect(self.update_stopfreq_value)

        self.rp.sweep_finished.connect(self.on_sweep_finished)
        
        # Connect live data chunk signal for live plotting
        self.rp.data_chunk_ready.connect(self.handle_data_chunk)

        self.open_device_connection()

    def closeEvent(self, event):
        if hasattr(self, 'rp') and self.rp is not None:
            self.rp.stop()
            print('RP disconnected')
        if hasattr(self, 'cryo') and self.cryo is not None:
            self.cryo.disconnect()
            print('Cryo disconnected')
        event.accept()

    def on_sweep_finished(self):
        if self.device_status['Cryocon'] == 'connected':
            self.t1 = self.cryo.get_temp('a')

        self.elapsed_time = np.round((time.time() - self.start_time), 5)

        self.progressBar.setValue(100)
        self.startButton.setEnabled(True)
        self.freq_raw = self.rp.get_reim().freq
        self.data_raw = self.rp.get_reim().data
        self.real_raw = np.real(self.data_raw)
        self.imag_raw = np.imag(self.data_raw)
        # self.real_data, self.imag_data, self.freq = self.interpolate_sweep(self.real_raw, self.imag_raw, self.freq_raw)
        self.real_data, self.imag_data, self.freq = self.real_raw, self.imag_raw, self.freq_raw
        self.plot_data()
        self.save_sweep()

        # update resonant frequencies by finding maximum in the window
        if self.tracking_sweeps.size != 0:
            mag = np.sqrt(self.real_data**2 + self.imag_data**2)
            resonance = []
            freqs = self.sweep_boundaries(self.tracking_sweeps)
            # print(freqs)
            for res in self.resonances_to_track:
                idx = np.searchsorted(freqs, res)
                start_idx = np.argmin(np.abs(self.freq - freqs[idx-1]))
                stop_idx = np.argmin(np.abs(self.freq - freqs[idx]))
                res_idx = np.argmax(mag[start_idx:stop_idx])
                resonance.append(self.freq[res_idx + start_idx])

            if len(resonance) == len(self.resonances_to_track):
                self.resonance_tracking.update_resonance(resonance)
            else:
                print('Resonance number does not match. Tracking Failed.')
            self.resonance_tracking.save_resonance_line()

        if self.continuousSweepBox.isChecked():
            self.current_sweep += 1
            self.start_sweep()
    
    # get the boundary points of the sweep frequency
    def sweep_boundaries(self, arr):
        starts = arr[:, 0]         
        last_end = arr[-1, 1]     
        return np.concatenate([starts, [last_end]]).flatten()
    
    def handle_data_chunk(self, chunk):
        # chunk: numpy array of N complex64 (N/2 samples, 2 channels interleaved)
        live_real = chunk[0::2].real
        live_imag = chunk[1::2].imag

        # Get current sweep parameters for this chunk
        # sweep_start = self.rp.get_sweep_start()
        # sweep_stop = self.rp.get_sweep_stop()
        # sweep_size = self.rp.get_sweep_size()

        self.live_real_accum.append(live_real)
        self.live_imag_accum.append(live_imag)

        self.real_data = np.concatenate(self.live_real_accum)
        self.imag_data = np.concatenate(self.live_imag_accum)
        length = len(self.real_data)
        self.freq = self.freq_points[:length]
        self.progressBar.setValue(length / len(self.freq_points) * 100)

        self.pw.clear()
        pen_real = pg.mkPen(color=(255, 0, 0), width=2)
        pen_imag = pg.mkPen(color=(0, 0, 255), width=2)
        self.pw.plot(x=self.freq, y=self.real_data, pen=pen_real, name='Real')
        self.pw.plot(x=self.freq, y=self.imag_data, pen=pen_imag, name='Imag')
    
    '''
    def handle_data_chunk(self, chunk):
        # chunk: numpy array of N complex64 (N/2 samples, 2 channels interleaved)
        live_real = chunk[0::2].real
        live_imag = chunk[1::2].imag

        # Accumulate all received data
        if not hasattr(self, 'live_real_accum'):
            self.live_real_accum = []
            self.live_imag_accum = []

        self.live_real_accum.append(live_real)
        self.live_imag_accum.append(live_imag)

        self.real_data = np.concatenate(self.live_real_accum)
        self.imag_data = np.concatenate(self.live_imag_accum)

        n_points = min(len(self.real_data), self.sweepsize)
        self.freq = np.linspace(self.startfreq, self.stopfreq, self.sweepsize)[:n_points]
        self.real_data = self.real_data[:n_points]
        self.imag_data = self.imag_data[:n_points]

        self.pw.clear()
        pen_real = pg.mkPen(color=(255, 0, 0), width=2)
        pen_imag = pg.mkPen(color=(0, 0, 255), width=2)
        self.pw.plot(x=self.freq, y=self.real_data, pen=pen_real, name='Real')
        self.pw.plot(x=self.freq, y=self.imag_data, pen=pen_imag, name='Imag')
    '''

    def openbinfile(self, filename):
        dat = open(filename, 'r')
        data = np.fromfile(dat, dtype=np.dtype('>f8'))  # the raw data are binary files, so this imports them
        split_index = int((len(data) - 1) / 3)
        frequency = data[1: split_index + 1]
        x = data[split_index + 1: 2 * split_index + 1]
        y = data[2 * split_index + 1: 3 * split_index + 1]
        return frequency, x, y
    
    def read_data(self, filename):
        initial_directory = fr'{Path(__file__).parent}'
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
            dat = np.loadtxt(filename, delimiter=',', dtype=float)
            self.freq_raw, self.real_raw, self.imag_raw = dat[:, 0], dat[:, 1], dat[:, 2]
        elif os.path.splitext(filename)[1] == '.bin':
            self.freq_raw, self.real_raw, self.imag_raw = self.openbinfile(filename)
        else:
            print(os.path.splitext(filename)[1])

        # self.real_data, self.imag_data, self.freq = self.interpolate_sweep(self.real_raw, self.imag_raw, self.freq_raw)
        self.real_data, self.imag_data, self.freq = self.real_raw, self.imag_raw, self.freq_raw

        self.plot_data()

    def save_dir(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if dir:
            self.save_path = dir
            self.saveDirLine.setText(dir)

    def save_sweep(self):
        f = self.freq
        real = self.real_data
        imag = self.imag_data
        try: self.temp = (self.t0 + self.t1) / 2
        except: self.temp = 0

        path = self.save_path if hasattr(self, 'save_dir') else Path(__file__).parent
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'{self.temp}K-{self.startfreq/1e6}M_{self.stopfreq/1e6}M-{int(len(real)/1e3)}k-{timestamp}.dat'
        filepath = os.path.join(path, filename)

        if filepath:
            fh = open(filepath, 'w')
            fh.write('#Lockin Constant: ' + str(self.rate).strip() + '\n')
            fh.write('#Data collection rate: ' + str(self.point_per_sec).strip() + ' points/s' +'\n')
            fh.write('#Temp at start of sweep: ' + str(self.t0).strip() + ' K' + '\n')
            fh.write('#Temp at end of sweep: ' + str(self.t1).strip() + ' K' +'\n')
            fh.write('#Average Temperature: ' + str(self.temp).strip() + ' K' +'\n')
            fh.write('#Sweep (MHz): ' + str(self.startfreq/1e6).strip() + ' to ' + str(self.stopfreq/1e6) +'\n')
            fh.write('#Step Size (Hz): ' + str(self.step).strip() + '\n')
            fh.write('#Sweep Size (k): ' + str(self.sweepsize/1e3).strip() + '\n')
            fh.write('#Drive Voltage (V): ' + str(self.rp.get_voltage()).strip() + '\n')
            fh.write('#Time Took (s): ' + str(self.elapsed_time).strip() + '\n')
            txt = str(self.current_sweep) if self.continuousSweepBox.isChecked() else 'No'
            fh.write('#Continuous Sweep: ' +  txt + '\n')
            fh.write('#Frequency (Hz), X (V), Y (V)\n')

            for i in range(len(f)):
                fh.write('%16.16f, %16.16f, %16.16f\n' % (f[i], real[i], imag[i]))
            fh.close()

    def plot_data(self):
        self.pw.clear()
        n = 10
        if self.current_filter == 1:
            data1, data2 =  self.real_data[n:],  self.imag_data[n:]
            data1 = data1 - np.mean(data1)
            data2 = data2 - np.mean(data2)
            data3 = np.sqrt(data1 ** 2 + data2 ** 2)
            frequency = self.freq[n:]
        else:
            data1, data2 = self.bandpass(self.real_data, self.imag_data, self.freq, self.lpcutoff, self.hpcutoff)
            data1, data2 = data1[n:], data2[n:]
            data1 = data1 - np.mean(data1)
            data2 = data2 - np.mean(data2)
            data3 = np.sqrt(data1 ** 2 + data2 ** 2)            
            frequency = self.freq[n:]

        styles = {'color': 'white', 'font-size': '30px'}

        if self.current_plot == 1:
            pen = pg.mkPen(color=(255, 0, 0), width=3, style=QtCore.Qt.SolidLine)
            self.pw.setLabel('left', 'Amplitude', units='V', **styles)
            self.pw.setLabel('bottom', 'Frequency', units='Hz', **styles)
            self.pw.setTitle("Magnitude", color='w', size='30px')
            self.pw.plot(y=data3, x=frequency, pen=pen)
        else:
            self.pw.setLabel('left', 'Amplitude', units='V', **styles)
            self.pw.setLabel('bottom', 'Frequency', units='Hz', **styles)
            self.pw.setTitle("Real / Img", color='w', size="30px")
            pen = pg.mkPen(color=(255, 0, 0), width=3, style=QtCore.Qt.SolidLine)
            self.pw.plot(y=data1, x=frequency, pen=pen)
            pen = pg.mkPen(color=(0, 0, 255), width=3, style=QtCore.Qt.SolidLine)
            self.pw.plot(y=data2, x=frequency, pen=pen)

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

    def create_tracking_sweeps(self, start_freq, stop_freq, resonances, tracking_window, dense_spacing, sparse_spacing):
        # print(
        #     f"Starting tracking sweep: {start_freq} - {stop_freq}, {resonances}, {tracking_window}, {dense_spacing}, {sparse_spacing}"
        # )

        start = float(min(start_freq, stop_freq))
        stop  = float(max(start_freq, stop_freq))
        tw = float(tracking_window)
        res = np.atleast_1d(resonances).astype(float)
        res = res[np.isfinite(res)]

        # Build initial dense intervals
        dense_intervals = []
        if res.size > 0 and tw > 0:
            for r in np.sort(res):
                a = max(start, r - tw/2.0)
                b = min(stop,  r + tw/2.0)
                if a < b:
                    dense_intervals.append([a, b])

            # Merge overlapping intervals
            dense_intervals.sort()
            merged = []
            for a, b in dense_intervals:
                if not merged or a > merged[-1][1]:
                    merged.append([a, b])
                else:
                    merged[-1][1] = max(merged[-1][1], b)
            dense_intervals = merged
        else:
            dense_intervals = []

        segments = []
        cursor = start
        for a, b in dense_intervals:
            if cursor < a:
                npts = int(np.floor((a - cursor) / sparse_spacing)) + 1
                segments.append([cursor, a, npts])
            npts = int(np.floor((b - a) / dense_spacing)) + 1
            segments.append([a, b, npts])
            cursor = b

        if cursor < stop:
            npts = int(np.floor((stop - cursor) / sparse_spacing)) + 1
            segments.append([cursor, stop, npts])

        return np.array(segments, dtype=float)
    
    def start_sweep(self):
        self.pw.clear()
        self.startButton.setEnabled(False)

        self.temp, self.t0, self.t1 = 0, 0, 0
        self.real_data = []
        self.imag_data = []
        self.freq = []
        self.live_real_accum = []
        self.live_imag_accum = []
        self.live_freq_accum = []
        self.tracking_sweeps = np.array([])

        self.rp.reset_reim()

        isReady = self.sweep_condition()

        if isReady == True:
            self.progressBar.setValue(0)
            self.rp.set_rate(self.rateValue.currentIndex())

            self.update_point_value()
            self.update_startfreq_value()
            self.update_stopfreq_value()
            self.update_volt_value()
            
            self.startfreq = self.rp.get_sweep_start()
            self.stopfreq = self.rp.get_sweep_stop()
            self.sweepsize = self.rp.get_sweep_size()
            self.rate = self.rp.get_rate()
            self.point_per_sec = int(self.rateValue.currentText())
            self.step = self.rp.get_stepValue()

            self.freq_points = np.linspace(self.startfreq, self.stopfreq, self.sweepsize)

            MAX_SWEEP_SIZE = getattr(self.rp, 'MAX_SWEEP_SIZE', 32766)

            if self.device_status['Cryocon'] == 'connected':
                self.t0 = self.cryo.get_temp('a')

            self.start_time = time.time()

            if self.resonanceTrackingBox.isChecked():
                self.tracking_sweeps = self.create_tracking_sweeps(
                    self.startfreq, 
                    self.stopfreq, 
                    self.resonances_to_track,
                    self.tracking_window, 
                    self.dense_spacing, 
                    self.sparse_spacing
                    )
                self.rp.start_tracking_sweep(self.tracking_sweeps)
                self.freq_points = [f for sweep in self.tracking_sweeps for f in np.linspace(sweep[0], sweep[1], int(sweep[2]))]
            elif self.sweepsize > MAX_SWEEP_SIZE:
                self.rp.start_long_sweep(self.startfreq, self.stopfreq, self.sweepsize)
            else:
                self.rp.sweep()
        else:
            QMessageBox.information(self, 'rus', 'Error: Sweep range or voltage is invalid.')
    
    def sweep_condition(self):
        if self.rp.get_sweep_start() < self.rp.get_sweep_stop():
            if self.rp.get_voltage() <= 2:
                return True
        return False
    
    def print_params(self):
        print(self.rp.get_stepValue())
        print(self.rp.get_sweep_start())
        print(self.rp.get_sweep_stop())
        print(self.rp.get_sweep_size())
        print(self.rp.get_voltage())

    def open_device_connection(self):
        if not hasattr(self, "device_connection") or not self.device_connection.isVisible():
            self.device_connection = deviceConnection(self.reactor, parent=self)
            self.device_connection.show()
        else:
            self.device_connection.raise_()
            self.device_connection.activateWindow()

    def open_resonance_detector(self):
        if not hasattr(self, "resonance_detector") or not self.resonance_detector.isVisible():
            self.resonance_detector = resonanceDetector(self.reactor, parent=self)
            self.resonance_detector.show()
        else:
            self.resonance_detector.raise_()
            self.resonance_detector.activateWindow()

    def open_resonance_tracking(self):
        if not hasattr(self, "resonance_tracking") or not self.resonance_tracking.isVisible():
            self.resonance_tracking = resonanceTracking(self.reactor, parent=self)
            self.resonance_tracking.show()
        else:
            self.resonance_tracking.raise_()
            self.resonance_tracking.activateWindow()


    def stop_sweep(self): 
        self.rp.cancel()
        self.startButton.setEnabled(True)
        self.current_sweep = 0

    def interpolate_sweep(self, real, img, freq, delta_f=None):
        f_min, f_max = np.min(freq), np.max(freq)
        df = np.median(freq[1:]-freq[:-1])
        if not delta_f is None: df = delta_f
        N = int((f_max-f_min)/df)
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
        
    def set_enabled(self, enabled):
        widgets = [self.rateValue, self.voltageValue, self.pointValue, self.startFreqValue,
                   self.stopFreqValue, self.startButton]
        for entry in widgets:
            entry.setEnabled(enabled)

    def update_volt_value(self):
        self.rp.set_volt1(self.voltageValue.value())
        self.voltageValue.setValue(self.rp.get_voltage())

    def update_point_value(self):
        self.rp.set_sweep_size(int(float(self.pointValue.text())*1e3))
        self.rp.set_step_value()
        self.stepValue.setText(self.rp.get_stepValue())

    def update_startfreq_value(self):
        self.rp.set_sweep_start(int(self.startFreqValue.value()*1e6))
        self.startFreqValue.setValue(self.rp.get_sweep_start()/1e6)
        self.rp.set_step_value()
        self.stepValue.setText(self.rp.get_stepValue())

    def update_stopfreq_value(self):
        self.rp.set_sweep_stop(int(self.stopFreqValue.value()*1e6))
        self.stopFreqValue.setValue(self.rp.get_sweep_stop()/1e6)
        self.rp.set_step_value()
        self.stepValue.setText(self.rp.get_stepValue())

    #async sleep function - GUI is operable while function sleeps
    def sleep(self, secs):
        d = Deferred()
        self.reactor.callLater(secs,d.callback,'Sleeping')
        return d

if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # window = MainWindow()
    # window.show()
    # arg = app.exec_()
    # window.disconnect()
    # sys.exit(arg)

    app = QApplication([])
    import qt5reactor
    qt5reactor.install()
    from twisted.internet import reactor
    window = MainWindow(reactor)
    sys.exit(app.exec_())