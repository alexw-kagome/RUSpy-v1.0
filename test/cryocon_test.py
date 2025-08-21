from cryocon_22C import Cryocon22C
import pyvisa
import time
# c = Cryocon22C(reactor=None, port='COM3')
# c.connect()
# print(c.id())
# print(c.set_unit('k', 'a'))
# print(c.get_temp('2b'))

rm = pyvisa.ResourceManager()
inst = rm.open_resource('COM3')
# print(inst.close())
print(inst.query('SYST:LOCK?'))
print(inst.query("SYSTem:LOCKout OFF"))
print(inst.query('SYST:LOCK?'))