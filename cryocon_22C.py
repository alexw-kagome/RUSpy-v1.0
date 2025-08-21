import numpy as np
import pyvisa
import twisted
from twisted.internet.defer import inlineCallbacks, Deferred
from PyQt5 import QtCore

# Simple temperature controller. Written by Alexander Won.
# https://file-storage.github.io/manuals/cryocon_22c_user_manual.pdf

class Cryocon22C(QtCore.QObject):
    connected_signal = QtCore.pyqtSignal(str)
    def __init__(self, reactor=None, port=None):
        super().__init__()
        self.reactor = reactor
        self.port = port
        self.rm = pyvisa.ResourceManager()
        self.inst = None

    def connect(self, port):
        try:
            self.port = port
            self.inst = self.rm.open_resource(self.port)
            self.connected_signal.emit('connected')
        except Exception as e:
            self.connected_signal.emit(str(e))

    def disconnect(self):
        if self.inst == None:
            return
        self.inst.close()
        self.inst = None

    def id(self):
        resp = self.inst.query("*IDN?")
        return resp
    
    def get_temp(self, channel):
        # lowercase
        # channel = str(channel).lower()
        resp = self.inst.query(f"input? {channel}")
        return resp

    def set_unit(self, unit, channel):
        # Choices are K- Kelvin, C- Celsius, F- Fahrenheit and S- native sensor units (Volts or Ohms)
        # unit = str(unit).lower()
        # channel = str(channel).lower()
        resp = self.inst.query(f"input {channel}: units {unit}")
        return resp
    
    #async sleep function - GUI is operable while function sleeps
    def sleep(self, secs):
        d = Deferred()
        self.reactor.callLater(secs,d.callback,'Sleeping')
        return d