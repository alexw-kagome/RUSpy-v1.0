import sys
import os
from matplotlib import text
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
from PyQt5.QtWidgets import QMainWindow, QTableWidget, QTableWidgetItem, QFileDialog, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QIcon
from pathlib import Path
from PyQt5 import uic
import time

class resonanceTracking(QMainWindow):
    variableChanged = pyqtSignal()
    def __init__(self, reactor, parent=None):
        super().__init__()
        base_dir = Path(__file__).parent
        ui_file = base_dir / 'resonance_tracking.ui'
        uic.loadUi(ui_file, self)

        logo_file = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(logo_file + os.path.sep + 'logo.png'))
        
        self.reactor = reactor
        self.parent = parent
        
        self.setWindowTitle('Resonance Tracking')

        self.loadpath = ''
        self.savepath = ''

        self.current_column = 0
        self.data = np.array([])

        self.loadButton.clicked.connect(self.load_resonances)
        self.saveButton.clicked.connect(self.create_savefile)
        self.denseSpacingLine.textChanged.connect(self.update_value)
        self.sparseSpacingLine.textChanged.connect(self.update_value)
        self.trackingWindowLine.textChanged.connect(self.update_value)
        self.clearTableButton.clicked.connect(self.clear_table)
        
        self.resonanceTable.setFocusPolicy(Qt.StrongFocus)
        self.resonanceTable.setEditTriggers(QTableWidget.NoEditTriggers)

        self.parent.dense_spacing = 10
        self.parent.sparse_spacing = 100
        self.parent.tracking_window = 3000

    def update_value(self):
        self.parent.dense_spacing = float(self.denseSpacingLine.text()) 
        self.parent.sparse_spacing = float(self.sparseSpacingLine.text()) 
        self.parent.tracking_window = float(self.trackingWindowLine.text())

    def load_resonances(self):
        # This is a good way to keep file directory. Patch this to other files if I am not lazy.
        base_dir = fr'{Path(__file__).parent}'
        if not self.loadpath:
            self.loadpath, _ = QFileDialog.getOpenFileName(self, 'Open file', base_dir, "Data Files (*.txt *.dat)")
        else:
            self.loadpath, _ = QFileDialog.getOpenFileName(self, 'Open file', self.loadpath,"Data Files (*.txt *.dat)")
        
        self.loadFileLine.setText(self.loadpath)

        try:
            self.data = np.loadtxt(self.loadpath, delimiter=',', dtype=float)
        except Exception as e:
            QMessageBox.information(self, 'Readfile', f'Error: {e}')
        
        self.current_column = 0 
        self.save_column = 0

        self.resonanceTable.clear()

        self.resonanceTable.setRowCount(self.data.shape[0])
        self.resonanceTable.setColumnCount(5000)
        self.resonanceTable.horizontalHeader().setDefaultSectionSize(300)
        self.resonanceTable.verticalHeader().setDefaultSectionSize(50)

        self.update_resonance(self.data[:, 0])

        # recommended tracking window = gamma * 2
        gamma = np.max(self.data[:, 1])
        width = int(gamma * 2)
        self.parent.tracking_window = width
        self.trackingWindowLine.setText(str(width))

        # self.resonanceTable.setHorizontalHeaderLabels(['Initial'])

    def update_resonance(self, data):
        data = np.sort(data)
        for i in range(len(data)):
            self.resonanceTable.setItem(i, self.current_column, QTableWidgetItem(str(data[i])))

        item = self.resonanceTable.item(0, self.current_column)
        self.resonanceTable.scrollToItem(item)

        self.parent.resonances_to_track = np.array(data).flatten()
        self.current_column += 1


    def clear_table(self):
        self.resonanceTable.clear()
        self.resonanceTable.setRowCount(0)
        self.resonanceTable.setColumnCount(0)
        self.current_column = 0
        self.save_column = 0

    def create_savefile(self):
        # This is a good way to keep file directory. Patch this to other files if I am not lazy.
        base_dir = fr'{Path(__file__).parent}'

        if not self.savepath:
            self.savepath = QFileDialog.getExistingDirectory(self, 'Select Folder', base_dir)
        else:
            self.savepath = QFileDialog.getExistingDirectory(self, 'Select Folder', self.savepath)
            
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'tracked_resonances_{timestamp}.dat'
        self.savepath = os.path.join(self.savepath, filename)
        self.saveFileLine.setText(self.savepath)

        with open(self.savepath, 'w') as file:
            file.write(f'#Dense Spacing: {self.parent.dense_spacing} Hz\n' + 
                       f'#Sparse Spacing: {self.parent.sparse_spacing} Hz\n' + 
                       f'#Tracking Window: {self.parent.tracking_window} Hz\n' +
                       f'#Temperature (K), [Resonances]\n')
            
        self.save_resonance_line()

    def save_resonance_line(self):
        rows = self.resonanceTable.rowCount()

        while self.save_column < self.current_column:
            with open(self.savepath, 'a') as file:
                file.write(f'{self.parent.temp}, ')
                res_line = ''

                for i in range(rows):
                    res = self.resonanceTable.item(i, self.save_column).text().strip()
                    res_line += res + ", "

                file.write(res_line[:-2] + "\n")
            self.save_column += 1

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
    window = resonanceTracking(reactor)
    sys.exit(app.exec_())