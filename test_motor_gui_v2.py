import os
import threading
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import motor_gui_v2 as gui
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class KeyboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.worker = Mock()
        self.worker.connected = threading.Event()
        settings = Mock()
        settings.value.side_effect = lambda key, default: default
        with patch.object(gui, "BleWorker", return_value=self.worker), \
                patch.object(QtCore, "QSettings", return_value=settings):
            self.window = gui.Window()
        self.window.timer.stop()
        self.window.status_timer.stop()
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        self.worker.connected.set()
        self.window.update()
        self.window.mode_timer.stop()
        self.worker.send.reset_mock()
        self.assertTrue(self.window.isActiveWindow())

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def press(self, key, repeat=False, modifiers=QtCore.Qt.NoModifier):
        editor = self.window.fasthome_voltage.lineEdit()
        editor.setFocus()
        event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, key, modifiers, "", repeat)
        self.app.sendEvent(editor, event)

    def commands(self):
        return [call.args[0] for call in self.worker.send.call_args_list]

    def test_disabled_by_default(self):
        self.assertFalse(self.window.enable_keys.isChecked())
        self.assertEqual(self.window.fasthome_voltage.value(), 1.0)
        self.assertEqual(self.window.fasthome_target.value(), 0.0)
        self.press(QtCore.Qt.Key_Return)
        self.press(QtCore.Qt.Key_H)
        self.assertEqual(self.commands(), [])

    def test_enter_matches_move_and_commits_edits(self):
        self.window.enable_keys.setChecked(True)
        self.window.setpoint.lineEdit().setText("4.5")
        self.window.target.lineEdit().setText("35.0")
        self.window.rebound_count = 2
        self.press(QtCore.Qt.Key_Return)
        expected = ["MODE VOLTAGE 4.5000", "MOVE 35.00000 2 2.00000 500"]
        self.assertEqual(self.commands(), expected)
        self.assertFalse(self.window.mode_timer.isActive())
        self.worker.send.reset_mock()
        self.window.move_button.click()
        self.assertEqual(self.commands(), expected)

    def test_fasthome_preserves_jump_settings(self):
        self.window.enable_keys.setChecked(True)
        self.window.setpoint.setValue(4.0)
        self.window.target.setValue(30.0)
        self.window.rebound_count = 2
        self.window.fasthome_voltage.lineEdit().setText("1.5")
        self.window.fasthome_target.lineEdit().setText("-3.5")
        self.press(QtCore.Qt.Key_H)
        self.assertEqual(self.commands(), ["MODE VOLTAGE 1.5000", "MOVE -3.50000 0"])
        self.assertFalse(self.window.mode_timer.isActive())
        self.assertTrue(self.window.data.recording)
        self.press(QtCore.Qt.Key_Return)
        self.assertEqual(self.commands()[-2:],
                         ["MODE VOLTAGE 4.0000", "MOVE 30.00000 2 2.00000 500"])

    def test_fasthome_uses_voltage_even_in_velocity_mode(self):
        self.window.enable_keys.setChecked(True)
        self.window.mode.setCurrentIndex(1)
        self.press(QtCore.Qt.Key_H)
        self.assertEqual(self.commands(), ["MODE VOLTAGE 1.0000", "MOVE 0.00000 0"])
        self.assertEqual(self.window.mode.currentText(), "Velocity")

    def test_held_and_modified_keys_do_not_move(self):
        self.window.enable_keys.setChecked(True)
        for key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_H):
            self.press(key, repeat=True)
            self.press(key, modifiers=QtCore.Qt.ControlModifier)
        self.assertEqual(self.commands(), [])

    def test_keypad_enter(self):
        self.window.enable_keys.setChecked(True)
        self.press(QtCore.Qt.Key_Enter, modifiers=QtCore.Qt.KeypadModifier)
        self.assertEqual(len(self.commands()), 2)

    def test_busy_and_disabled_move_block_keys(self):
        self.window.enable_keys.setChecked(True)
        for state in (1, 2, 3):
            self.window.data.state = state
            self.press(QtCore.Qt.Key_Return)
            self.press(QtCore.Qt.Key_H)
        self.window.data.state = 0
        self.window.move_button.setEnabled(False)
        self.press(QtCore.Qt.Key_Return)
        self.press(QtCore.Qt.Key_H)
        self.assertEqual(self.commands(), [])

    def test_disconnect_blocks_keys_and_disarms(self):
        self.window.enable_keys.setChecked(True)
        self.worker.connected.clear()
        self.press(QtCore.Qt.Key_Return)
        self.press(QtCore.Qt.Key_H)
        self.window.move()
        self.window.fasthome()
        self.assertEqual(self.commands(), [])
        self.window.update()
        self.assertFalse(self.window.enable_keys.isChecked())
        self.worker.connected.set()
        self.window.update()
        self.assertFalse(self.window.enable_keys.isChecked())

    def test_focus_loss_disarms(self):
        self.window.enable_keys.setChecked(True)
        self.app.sendEvent(self.window, QtCore.QEvent(QtCore.QEvent.WindowDeactivate))
        self.assertFalse(self.window.enable_keys.isChecked())
        self.press(QtCore.Qt.Key_H)
        self.assertEqual(self.commands(), [])

    def test_modal_dialog_blocks_keys(self):
        self.window.enable_keys.setChecked(True)
        dialog = QtWidgets.QDialog(self.window)
        dialog.setModal(True)
        dialog.show()
        self.app.processEvents()
        self.press(QtCore.Qt.Key_Return)
        self.press(QtCore.Qt.Key_H)
        self.assertEqual(self.commands(), [])
        dialog.close()

    def test_fasthome_over_battery_voltage_does_not_move(self):
        self.window.enable_keys.setChecked(True)
        self.window.data.samples.append((0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0))
        self.press(QtCore.Qt.Key_H)
        self.assertEqual(self.commands(), [])
        self.assertIn("battery voltage", self.window.error)


if __name__ == "__main__":
    unittest.main()
