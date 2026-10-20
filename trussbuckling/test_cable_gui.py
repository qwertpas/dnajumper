"""Exercise the cord viewer's user workflow without opening a desktop window."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from trussbuckling.cable_gui import CableWindow


class CableGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_presets_sensitivity_limits_and_invalid_load(self):
        window = CableWindow()
        try:
            window.show()
            self.app.processEvents()
            self.assertEqual(window._result[1].points.shape, (13, 6, 3))
            self.assertTrue(window._result[1].stable)
            self.assertIn("10% R", window.caption.text())

            window.preset.setCurrentIndex(1)
            window.refresh()
            self.assertEqual(window._result[1].points.shape, (13, 8, 3))

            window.view.setCurrentIndex(1)
            window.refresh()
            self.assertFalse(window.controls["amplitude"].isEnabled())
            self.assertTrue(window.controls["offset"].isEnabled())
            self.assertIn("no visual amplification", window.caption.text())
            self.assertEqual(len(window.tilt_axis.lines), 3)
            self.assertNotIn("slacken", window.caption.text())
            window.controls["offset"].number.setValue(.1)
            window.controls["tilt"].number.setValue(.1)
            window.refresh()
            self.assertIn("slacken", window.caption.text())
            self.assertIn("Single cell:", window.status.text())

            window.preset.setCurrentIndex(2)
            window.refresh()
            self.assertFalse(window._result[1].tilt_stable)
            self.assertEqual(window.controls["height"].value(), 1.414)
            self.assertEqual(window.controls["twist"].value(), 45)
            self.assertIn("No stable linear response", window.status.text())
            self.assertEqual(len(window.tilt_axis.lines), 1)

            window.view.setCurrentIndex(0)
            window.refresh()
            self.assertEqual(len(window.tilt_axis.lines), 3)
            self.assertTrue(window.controls["amplitude"].isEnabled())

            window.controls["twist"].number.setValue(75)
            window.controls["height"].number.setValue(.3)
            window.refresh()
            self.assertLess(window.controls["twist"].value(), 20)
            self.assertGreaterEqual(window._result[1].height, 1.9)
            self.assertNotIn("Invalid settings", window.status.text())

            window.reset()
            window.controls["rigidity"].number.setValue(10)
            window.controls["force"].number.setValue(1000)
            window.refresh()
            self.assertIn("Invalid settings", window.status.text())
            self.assertIn("previous drawing", window.status.text())
            window.reset()
            self.app.processEvents()
            self.assertIn("Stable stack", window.status.text())
            self.assertEqual(window.controls["twist"].value(), 45)
            self.assertEqual(window.controls["offset"].value(), .01)
            self.assertEqual(window.controls["tilt"].value(), .01)
            self.assertEqual(window._result[1].points.shape, (13, 6, 3))
        finally:
            window.timer.stop()
            window.close()


if __name__ == "__main__":
    unittest.main()
