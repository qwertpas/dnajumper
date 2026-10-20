"""Launch each GUI with PyQt6, render a frame, and close without using hardware."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parent
LAUNCH = """
import importlib
import sys
import tempfile
import threading
from unittest.mock import patch

module = importlib.import_module(sys.argv[1])
sys.argv = [module.__file__, *sys.argv[2:]]

from PyQt6 import QtCore, QtWidgets

errors = []
sys.excepthook = lambda *error: errors.append(error)
run = QtWidgets.QApplication.exec

def close():
    try:
        windows = [w for w in QtWidgets.QApplication.topLevelWidgets() if w.isVisible()]
        assert windows, 'No visible window'
        for window in windows:
            assert not window.grab().isNull(), 'Window did not render'
            window.close()
        assert 'PyQt5.QtCore' not in sys.modules
        assert 'PySide6.QtCore' not in sys.modules
        if 'pyqtgraph.Qt' in sys.modules:
            assert sys.modules['pyqtgraph.Qt'].QT_LIB == 'PyQt6'
        if 'matplotlib.backends.qt_compat' in sys.modules:
            assert sys.modules['matplotlib.backends.qt_compat'].QT_API == 'PyQt6'
    except Exception:
        errors.append(sys.exc_info())
    finally:
        QtWidgets.QApplication.quit()

def event_loop(self):
    QtCore.QTimer.singleShot(200, close)
    result = run()
    if errors:
        raise errors[0][1].with_traceback(errors[0][2])
    print('PyQt6 rendered and closed', flush=True)
    return result

with tempfile.TemporaryDirectory() as folder:
    if module.__name__ == 'new_modeling.springchain':
        module.__file__ = folder + '/springchain.py'
    with patch.object(threading.Thread, 'start'), patch.object(threading.Thread, 'join'), \
            patch.object(QtWidgets.QApplication, 'exec', event_loop):
        module.main()
"""


class GuiLaunchTests(unittest.TestCase):
    def test_launches(self):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        for name in ("PYQTGRAPH_QT_LIB", "QT_API", "MPLBACKEND"):
            env.pop(name, None)
        for module, args in (
            ("motor_gui_v2", []),
            ("motor_mindaq_gui", []),
            ("ft_live_gui", []),
            ("ble_stream_plot", []),
            ("trussbuckling.gui", []),
            ("trussbuckling.cable_gui", []),
            ("trussbuckling.fabric_gui", []),
            ("trussbuckling.research.no_shear.gui", []),
            ("trussbuckling.research.rigid_demo", ["--sides", "3"]),
            ("new_modeling.springchain", ["--show"]),
        ):
            with self.subTest(gui=module):
                result = subprocess.run(
                    [sys.executable, "-c", LAUNCH, module, *args],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PyQt6 rendered and closed", result.stdout)


if __name__ == "__main__":
    unittest.main()
