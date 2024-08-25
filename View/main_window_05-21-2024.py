import sys
from Controller.rus import rus
import os
import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import QSettings, QTimer
from PyQt5.QtNetwork import QTcpSocket
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QSizePolicy, QErrorMessage
from pathlib import Path
from PyQt5 import uic

# from PyQt5.QtCore import Null, QRegExp, QTimer, QSettings, QDir, Qt
from PyQt5.QtGui import QRegExpValidator, QPalette, QColor, QBitmap, QPixmap, QFont
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import (QApplication, QDoubleSpinBox, QMainWindow, QMessageBox, QDialog, QFileDialog, QPushButton,
                             QLabel, QSpinBox)
# from PyQt5.QtNetwork import QAbstractSocket, QTcpSocket
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplashScreen, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        base_dir = Path(__file__).parent
        print(base_dir)
        ui_file = base_dir / "main_ui.ui"
        uic.loadUi(ui_file, self)

        self.setWindowTitle("RUS")
        self.showMaximized()

        self.exp = rus()
        self.exp.set_addressValue("10.84.241.73")
        self.addressValue.setText(self.exp.get_addressValue())
        # settings = QSettings("rus.ini", QSettings.IniFormat)
        # self.read_cfg_settings(settings)

        # Plot GUI
        self.layout = QVBoxLayout(self.plot_widget)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Create a toolbar and add it to the layout
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.toolbar.setStyleSheet("background-color: #FFFFFF")
        self.toolbar.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        toolbar_button_layout = QHBoxLayout()
        toolbar_button_layout.addWidget(self.toolbar)

        # Create the button
        self.plotButton = QPushButton("Magnitude", self)
        self.plotButton.setStyleSheet("background-color: #219EBC")
        self.plotButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.plotButton.setFixedSize(400, 45)
        self.plotButton.setText("See Magnitude")
        toolbar_button_layout.addWidget(self.plotButton)
        font = QFont()
        font.setPointSize(15)
        font.setBold(True)
        self.plotButton.setFont(font)

        self.layout.addLayout(toolbar_button_layout)

        self.ax = self.fig.add_subplot(111)
        self.current_plot = 2
        self.data_x = 0
        self.data_y1 = 0
        self.data_y2 = 0
        self.plot_data()

        # configure widgets
        self.pointValue.setReadOnly(True)
        # self.rateValue.addItems([ "1500","500", "150", "50", "15", "5", "2"])
        self.rateValue.addItems(["1500", "500", "150", "50"])
        # self.rateValue.addItems(["5000", "1000", "500", "100", "50", "10", "5", "1"])
        # rate = [10, 50, 100, 500, 1000, 5000, 10000, 50000][value]
        self.set_enabled(False)

        self.progressBar.setValue(0)
        self.pbarVal = 0
        self.pbarStep = 0
        self.sweepsize = 0
        self.rate = 0
        self.delay = 0
        self.const = 0
        self.num = 0
        self.count = 0

        # connect signals from widgets
        self.plotButton.clicked.connect(self.switch_plot)
        self.connectButton.clicked.connect(self.connectButtonClicked)
        self.saveDatButton.clicked.connect(self.save)
        self.startButton.clicked.connect(self.startSweep)
        self.stopButton.clicked.connect(self.stopSweep)
        self.addressValue.textChanged.connect(self.exp.set_addressValue)

        self.rateValue.currentIndexChanged.connect(self.exp.set_rate)

        self.voltageValue.valueChanged.connect(self.updateVoltValue)
        self.stepValue.valueChanged.connect(self.updateStepValue)
        self.startFreqValue.valueChanged.connect(self.updateStartFreqValue)
        self.stopFreqValue.valueChanged.connect(self.updateStopFreqValue)

        # create timers
        self.connectTimer = QTimer()
        self.connectTimer.timeout.connect(self.check_connection)

        self.sweepTimer = QTimer()
        self.sweepTimer.timeout.connect(self.check_sweep)

        self.pbarTimer = QTimer()
        self.pbarTimer.setSingleShot(True)
        self.pbarTimer.timeout.connect(self.update_pbar)

    def plot_data(self):
        self.ax.clear()
        freq = self.exp.get_reim().freq
        # sstep = (freqP[2]-freqP[1])
        data = self.exp.get_reim().data
        data1 = np.real(data)
        data2 = np.imag(data)
        data3 = np.sqrt(data1 ** 2 + data2 ** 2)
        if self.current_plot == 1:
            self.ax.plot(freq, data3)
            self.ax.set_title("Magnitude", fontsize=20)
            self.ax.set_xlabel("Frequency (kHz)", fontsize=20)
            self.ax.set_ylabel("Magnitude", fontsize=20)
        else:
            self.ax.plot(freq, data1)
            self.ax.plot(freq, data2)
            self.ax.set_title("Real/Img", fontsize=20)
            self.ax.set_xlabel("Frequency (kHz)", fontsize=20)
            self.ax.set_ylabel("Magnitude", fontsize=20)
        self.canvas.draw()

    def switch_plot(self):
        # if self.exp.idle
        if self.current_plot == 1:
            self.current_plot = 2
            self.plotButton.setText("See Magnitude")
        else:
            self.current_plot = 1
            self.plotButton.setText("See Real/Img")
        self.plot_data()

    def timeout(self):
        self.display_error("timeout")

    def display_error(self, socketError):
        self.startTimer.stop()
        if socketError == "timeout":
            QMessageBox.information(self, "rus", "Error: connection timeout.")
        else:
            QMessageBox.information(self, "rus", "Error: %s." % self.socket.errorString())
        self.stop()

    def startSweep(self):
        isReady = self.sweepCondition()
        if isReady == True:
            self.progressBar.setValue(0)
            self.exp.set_rate(self.rateValue.currentIndex())
            self.exp.set_volt1(self.voltageValue.value())
            self.exp.set_stepValue(self.stepValue.value())
            self.exp.set_sweep_start(self.startFreqValue.value())
            self.exp.set_sweep_stop(self.stopFreqValue.value())
            self.set_enabled(False)

            self.printParams()
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
            self.num = 1000
            self.delay = 1000 * self.sweepsize / self.const / self.num
            self.progressBar.setMaximum(self.num)
            self.pbarStep = 1
            self.pbarVal = 0
            self.count = 0

            self.sweepTimer.start(100)
            self.pbarTimer.start(self.delay)
            self.exp.sweep()
        else:
            QMessageBox.information(self, "rus", "Error: Sweep size exceeds 32766 or incorrect sweep range.")
            # self.sweepTimer.start(1000)
    
    def sweepCondition(self):
        if self.exp.get_sweep_start() < self.exp.get_sweep_stop():
            if self.exp.get_sweep_size() < 32766:
                if self.exp.get_voltage() <= 2:
                    return True
        return False
    
    def printParams(self):
        print(self.exp.get_stepValue())
        print(self.exp.get_sweep_start())
        print(self.exp.get_sweep_stop())
        print(self.exp.get_sweep_size())
        print(self.exp.get_voltage())

    # def update_pbar(self):
    #     if self.oldcount != self.exp.get_count():
    #         # print('update pbar')
    #         self.pbarVal = self.pbarVal + 225000 / self.sweepsize / self.rate
    #         self.progressBar.setValue(self.pbarVal)
    #         self.oldcount = self.exp.get_count()

    def update_pbar(self):
        if self.num > self.count:
            # print('update pbar')
            self.pbarVal = self.pbarVal + self.pbarStep
            self.progressBar.setValue(self.pbarVal)
            self.count = self.count + 1
            self.pbarTimer.start(self.delay)



    def check_sweep(self):
        if self.exp.get_reading() == False:
            if self.exp.get_elapsedTime() != 0:
                self.sweepTimer.stop()
                self.pbarTimer.stop()
                self.progressBar.setValue(self.num)
                self.plot_data()
                self.set_enabled(True)

    def stopSweep(self): 
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

    def connectButtonClicked(self):
        if self.connectButton.text() == "Connect":
            self.disconnect()
            self.exp.start(self.exp.get_addressValue())
            self.connectTimer.start(100)
        else:
            self.disconnect()
            self.exp.set_connection_status("")
            self.set_enabled(False)
            self.connectButton.setText("Connect")

    def check_connection(self):
        if self.exp.get_connect_status() != "":
            self.connectTimer.stop()
            if self.exp.get_connect_status() == "connected":
                self.connectButton.setText("Disconnect")
                self.connectButton.setEnabled(True)
                self.set_enabled(True)
                print("CONNECTED")
                return
            if self.exp.get_socketError() == "timeout":
                QMessageBox.information(self, "rus", "Error: connection timeout.")
                self.exp.set_connection_status("")
            else:
                QMessageBox.information(self, "rus", "Error: %s." % self.exp.get_connect_status())
                self.exp.set_connection_status("")
    
    def set_enabled(self, enabled):
        widgets = [self.rateValue, self.voltageValue, self.pointValue, self.stepValue, self.startFreqValue,
                   self.stopFreqValue, self.startButton]
        for entry in widgets:
            entry.setEnabled(enabled)

    def updateVoltValue(self):
        self.exp.set_volt1(self.voltageValue.value())
        self.voltageValue.setValue(self.exp.get_voltage())

    def updateStepValue(self):
        self.exp.set_stepValue(self.stepValue.value())
        self.exp.set_sweep_size()
        self.stepValue.setValue(self.exp.get_stepValue())
        self.pointValue.setValue(self.exp.get_sweep_size())

    def updateStartFreqValue(self):
        self.exp.set_sweep_start(self.startFreqValue.value())
        self.startFreqValue.setValue(self.exp.get_sweep_start())
        self.exp.set_sweep_size()
        self.stepValue.setValue(self.exp.get_stepValue())
        self.pointValue.setValue(self.exp.get_sweep_size())

    def updateStopFreqValue(self):
        self.exp.set_sweep_stop(self.stopFreqValue.value())
        self.stopFreqValue.setValue(self.exp.get_sweep_stop())
        self.exp.set_sweep_size()
        self.stepValue.setValue(self.exp.get_stepValue())
        self.pointValue.setValue(self.exp.get_sweep_size())

    def save(self):
        path = os.getcwd()
        name = os.path.join(path, f'{self.exp.get_rate()}_rate_{self.exp.get_stepValue()}_step_{self.exp.get_sweep_size()}_sweepsize.dat')
        filename, _ = QFileDialog.getSaveFileName(self, "Save Data", name, "Data (*.dat)")
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
                fh.write("#  frequency        real         imag\n")
                for i in range(len(f)):
                    fh.write("%12.5f  %12.10f  %12.10f\n" % (f[i], d.real[i], d.imag[i]))
                fh.close()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    arg = app.exec_()
    window.disconnect()
    print('disconnected')
    sys.exit(arg)


if __name__ == "__main__":
    main()
