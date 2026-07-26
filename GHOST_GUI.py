#!/usr/bin/env python3
"""
GHOST -- the geometry / solver GUI
==================================

Draw and inspect a .geo, then run a solve on it, without writing a script.  It
is a separate way in to the SAME Backend the numbered steps use -- not a step
of its own, which is why it sits beside them rather than among them.

Use it to check a drawing before committing a long sweep to 1a/1b or 2; use the
numbered steps for anything you want repeatable.

    python3 GHOST_GUI.py

Requires PySide6 (the numbered steps do not).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Backend"))


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

    from geometry_tab import GeometryTab
    from solver_tab import SolverTab

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("GHOST")
            tabs = QTabWidget()
            self.geometry_tab = GeometryTab()
            self.solver_tab = SolverTab(geometry_tab=self.geometry_tab)
            tabs.addTab(self.geometry_tab, "Geometry")
            tabs.addTab(self.solver_tab, "Solver")
            self.setCentralWidget(tabs)
            self.resize(1000, 600)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


def main() -> int:
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
