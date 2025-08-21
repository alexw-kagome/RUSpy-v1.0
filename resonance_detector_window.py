import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import time
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from datetime import datetime

from lorentzian_fitting import FitResonances

import twisted
from twisted.internet.defer import inlineCallbacks, Deferred

from PyQt5.QtCore import QSettings, QTimer, Qt, QThread, pyqtSignal, QObject, QMutex, QMutexLocker
from PyQt5.QtNetwork import QTcpSocket
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QFileDialog, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QIcon
from pathlib import Path
from PyQt5 import uic
import time

class resonanceDetector(QWidget):
    variableChanged = pyqtSignal()
    def __init__(self, reactor, parent=None):
        super().__init__()
        base_dir = Path(__file__).parent
        ui_file = base_dir / 'resonance_detector.ui'
        uic.loadUi(ui_file, self)

        logo_file = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(logo_file + os.path.sep + 'logo.png'))
        
        self.reactor = reactor
        self.parent = parent
        
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

        self.resultTable.horizontalHeader().setDefaultSectionSize(300)
        self.resultTable.verticalHeader().setDefaultSectionSize(50)

        self.saveTable.setRowCount(500)
        self.saveTable.setColumnCount(9)
        self.saveTable.setHorizontalHeaderLabels(['f0', 'Gamma', 'Q', 'A', 'Phi',
                                                'X background intercept',
                                                'X background slope', 
                                                'Y background intercept',
                                                'Y background slope'])
        
        self.saveTable.horizontalHeader().setDefaultSectionSize(300)
        self.saveTable.verticalHeader().setDefaultSectionSize(50)

        self.resultTable.setFocusPolicy(Qt.StrongFocus)
        self.resultTable.setEditTriggers(QTableWidget.NoEditTriggers)
        
        self.saveTable.setFocusPolicy(Qt.StrongFocus)
        self.saveTable.setEditTriggers(QTableWidget.NoEditTriggers)

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
        self.loadTableButton.clicked.connect(self.load_table)

    def browsefiles(self):
        base_dir = fr'{Path(__file__).parent}'
        fname = QFileDialog.getOpenFileName(self, 'Open file', base_dir)
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
            dat = np.loadtxt(self.filepath, delimiter=',', dtype=float)
            self.freq_raw, self.real_raw, self.imag_raw = dat[:, 0], dat[:, 1], dat[:, 2]
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
        nyq = 2 * nyq_low * df
        fb, fa = butter(3, nyq, btype= 'hp', analog= False)
        Xlp = filtfilt(fb, fa, X)

        # low pass filter after high pass filter
        nyq = 2 * nyq_high * df
        if nyq >= 1:
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
                # self.resultTable.setItem(index, 0, QTableWidgetItem(str(np.round(value, 5))))
                self.resultTable.setItem(index, 0, QTableWidgetItem(str(value)))
        else:
            for index, value in enumerate(data):
                self.saveTable.setItem(self.current_row, index, QTableWidgetItem(str(value)))
            self.move_row(1)      
            self.highlight_row()   

    def remove_row(self):
        if self.saveTable.rowCount() > 0 and self.current_row < self.saveTable.rowCount():
            self.saveTable.removeRow(self.current_row)
            self.highlight_row()

        # If all rows are deleted, reset the table
        if self.saveTable.rowCount() == 0:
            self.reset_save_table()

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

    def reset_save_table(self):
        self.saveTable.setRowCount(500)
        self.saveTable.setColumnCount(9)
        self.saveTable.setHorizontalHeaderLabels([
            'f0', 'Gamma', 'Q', 'A', 'Phi',
            'X background intercept',
            'X background slope',
            'Y background intercept',
            'Y background slope'])
        
    def save_table(self):
        initial_directory = os.getcwd()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f'resonances_{timestamp}.dat'
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
                header_line = '#' + ', '.join(headers) + '\n'
                file.write(header_line)
                for row in range(self.saveTable.rowCount()):
                    items = [self.saveTable.item(row, col).text() if self.saveTable.item(row, col) else ''
                             for col in range(self.saveTable.columnCount())]
                    
                    # Only write the row if at least one cell is not empty
                    if any(cell.strip() for cell in items):
                        line = ', '.join(items)
                        file.write(line + '\n')

    def load_table(self):
        base_dir = fr'{Path(__file__).parent}'
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            'Open file',
            base_dir,
            "Data Files (*.txt *.dat)"
        )

        try:
            data = np.loadtxt(filepath, delimiter=',', dtype=float)
        except Exception as e:
            QMessageBox.information(self, 'Readfile', f'Error: {e}')
        
        self.reset_save_table()

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data[i, j]))
                self.saveTable.setItem(i, j, item)

    def plot_fit(self):
        mask = (self.freq >= self.last_position[self.vline1]) & (self.freq <= self.last_position[self.vline2])
        data1 = self.data1[mask]
        data2 = self.data2[mask]
        freq = self.freq[mask]

        if len(freq) != 0:
            fit = FitResonances(freq, data1, data2)
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
            spacing = self.vline2.value() - self.vline1.value()
            res = float(str(self.saveTable.item(self.current_row, 0).text()))
            self.vline1.setValue(res - spacing / 2)
            self.vline2.setValue(res + spacing / 2)
        elif key == Qt.Key_PageDown:
            self.move_row(1)
            spacing = self.vline2.value() - self.vline1.value()
            res = float(str(self.saveTable.item(self.current_row, 0).text()))
            self.vline1.setValue(res - spacing / 2)
            self.vline2.setValue(res + spacing / 2)
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

    #async sleep function - GUI is operable while function sleeps
    def sleep(self, secs):
        d = Deferred()
        self.reactor.callLater(secs,d.callback,'Sleeping')
        return d

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    import qt5reactor
    qt5reactor.install()
    from twisted.internet import reactor
    window = resonanceDetector(reactor)
    sys.exit(app.exec_())