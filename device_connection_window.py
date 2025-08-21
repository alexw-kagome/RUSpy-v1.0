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
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtGui import QIcon
from pathlib import Path
from PyQt5 import uic
import time

class deviceConnection(QMainWindow):

    def __init__(self, reactor, parent):
        super().__init__()
        base_dir = Path(__file__).parent
        ui_file = base_dir / 'device_connection.ui'
        uic.loadUi(ui_file, self)

        logo_file = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(logo_file + os.path.sep + 'logo.png'))
        self.setWindowTitle('Device Connection')

        self.reactor = reactor
        self.parent = parent

        self.parent.rp.connected_signal.connect(self.on_rp_connected)
        self.parent.cryo.connected_signal.connect(self.on_cryo_connected)

        self.rpAddressLine.setText(self.parent.device_address['Red Pitaya'])
        self.cryoAddressLine.setText(self.parent.device_address['Cryocon'])

        self.rpConnectButton.clicked.connect(self.rp_connect)
        self.cryoConnectButton.clicked.connect(self.cryo_connect)

        self.rp_status = ''
        self.cryo_status = ''

        if self.parent.device_status['Cryocon'] == 'connected':
            self.cryoConnectButton.setText('Disconnect')
        if self.parent.device_status['Red Pitaya'] == 'connected':
            self.rpConnectButton.setText('Disconnect')

    def on_rp_connected(self, status):
        self.parent.device_status['Red Pitaya'] = status
        self.rp_status = status
        if status == 'connected':
            self.rpConnectButton.setText('Disconnect')
            self.parent.set_enabled(True)
        elif status == 'timeout':
            QMessageBox.information(self, 'Red Pitaya', 'Error: connection timeout.')
        else:
            QMessageBox.information(self, 'Red Pitaya', 'Error: %s.' % status)

    def rp_connect(self):
        if self.rpConnectButton.text() == 'Connect':
            self.parent.rp.start(self.rpAddressLine.text())
        else:
            self.parent.rp.stop()
            self.parent.rp.set_connection_status('')
            self.parent.set_enabled(False)
            self.rpConnectButton.setText('Connect')
            self.parent.device_status['Red Pitaya'] = None

    def on_cryo_connected(self, status):
        self.parent.device_status['Cryocon'] = status
        self.cryo_status = status
        if status == 'connected':
            self.cryoConnectButton.setText('Disconnect')
            self.update_cryo_temp()
        else:
            QMessageBox.information(self, 'Cryocon', 'Error: %s.' % status)
    
    def cryo_connect(self):
        if self.cryoConnectButton.text() == 'Connect':
            self.parent.cryo.disconnect()
            self.parent.cryo.connect(port=self.cryoAddressLine.text())
        else:
            self.parent.cryo.disconnect()
            self.cryoConnectButton.setText('Connect')
            self.parent.device_status['Cryocon'] = None

    @inlineCallbacks
    def update_cryo_temp(self):
        while self.cryo_status == 'connected':
            temp = self.parent.cryo.get_temp('a')
            self.cryoTemp.setText(f'{temp}')
            yield self.sleep(1)

    #async sleep function - GUI is operable while function sleeps
    def sleep(self, secs):
        d = Deferred()
        self.reactor.callLater(secs,d.callback,'Sleeping')
        return d