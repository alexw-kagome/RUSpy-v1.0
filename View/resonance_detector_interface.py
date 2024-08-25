import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 800, 600)

        # Create the PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # Generate some sample data
        self.x = range(100)
        self.y = [i**0.5 for i in self.x]  # Sample data

        # Plot the data
        self.plot_widget.plot(self.x, self.y, pen=None, symbol='o')

        # Initialize vertical lines and movable dot
        self.vline1 = pg.InfiniteLine(pos=20, angle=90, movable=True, pen=pg.mkPen('w'))
        self.vline2 = pg.InfiniteLine(pos=30, angle=90, movable=True, pen=pg.mkPen('w'))

        # Add items to the plot
        self.plot_widget.addItem(self.vline1)
        self.plot_widget.addItem(self.vline2)

        # Store initial positions
        self.last_position = {self.vline1: self.vline1.value(), self.vline2: self.vline2.value()}
        self.syncing = False

        # Connect the signal for moving the lines
        self.vline1.sigPositionChanged.connect(self.sync_lines)
        self.vline2.sigPositionChanged.connect(self.sync_lines)

    def sync_lines(self):
        
        if self.syncing:
            return
        
        print('syncing')
        self.syncing = True

        # Get the current positions of the vertical lines
        pos1 = self.vline1.value()
        pos2 = self.vline2.value()

        # Determine the last known positions
        last_pos1 = self.last_position[self.vline1]
        last_pos2 = self.last_position[self.vline2]

        # Determine the displacement
        if pos1 != last_pos1:
            # vline1 was moved
            displacement = pos1 - last_pos1
            self.vline2.sigPositionChanged.disconnect(self.sync_lines)  # Temporarily disconnect
            self.vline2.setValue(last_pos2 + displacement)  # Move the second line
            self.vline2.sigPositionChanged.connect(self.sync_lines)  # Reconnect
        else:
            # vline2 was moved
            displacement = pos2 - last_pos2
            self.vline1.sigPositionChanged.disconnect(self.sync_lines)  # Temporarily disconnect
            self.vline1.setValue(last_pos1 + displacement)  # Move the first line
            self.vline1.sigPositionChanged.connect(self.sync_lines)  # Reconnect

        # Update the last positions
        self.last_position[self.vline1] = self.vline1.value()
        self.last_position[self.vline2] = self.vline2.value()

        self.syncing = False

    def keyPressEvent(self, event):
        self.syncing = True

        key = event.key()
        if key == QtCore.Qt.Key_Up:
            self.adjust_range(1)
        elif key == QtCore.Qt.Key_Down:
            self.adjust_range(-1)
        elif key == QtCore.Qt.Key_Left:
            self.move_lines(-1)
        elif key == QtCore.Qt.Key_Right:
            self.move_lines(1)

        # Update the last positions
        self.last_position[self.vline1] = self.vline1.value()
        self.last_position[self.vline2] = self.vline2.value()

        self.syncing = False
        

    def adjust_range(self, delta):
        v1_pos = self.vline1.value()
        v2_pos = self.vline2.value()

        if v2_pos - v1_pos > - 2 * delta:  # Ensure some minimum range
            self.vline1.setValue(v1_pos - delta)
            self.vline2.setValue(v2_pos + delta)


    def move_lines(self, delta):
        self.vline1.setValue(self.vline1.value() + delta)
        self.vline2.setValue(self.vline2.value() + delta)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
