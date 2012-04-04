#make the program quit on Ctrl+C
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from PyQt4.QtGui import QApplication

from ilastikshell.ilastikShell import IlastikShell
from applets.pixelClassification import PixelClassificationApplet

app = QApplication([])

#g = Graph()
#pipeline = PixelClassificationPipeline( g )
#pig = PixelClassificationGui( pipeline, graph = g)
#pig.show()

pc = PixelClassificationApplet()

from ilastikshell.applet import Applet
from PyQt4.QtGui import QMenuBar, QMenu
defaultApplet = Applet()
# Normally applets would provide their own menu items,
# but for this test we'll add them here (i.e. from the outside).
defaultApplet._menuWidget = QMenuBar()
defaultApplet._menuWidget.setNativeMenuBar( False ) # Native menus are broken on Ubuntu at the moment
defaultMenu = QMenu("Default Applet", defaultApplet._menuWidget)
defaultMenu.addAction("Default Action 1")
defaultMenu.addAction("Default Action 2")
defaultApplet._menuWidget.addMenu(defaultMenu)

shell = IlastikShell()
shell.addApplet(pc)
shell.addApplet(defaultApplet)
shell.show()

app.exec_()
