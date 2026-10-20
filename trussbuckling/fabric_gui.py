"""Standalone six-cell fabric assembly viewer: python3.11 -m trussbuckling.fabric_gui."""
import sys

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                            QVBoxLayout, QLabel, QSlider, QCheckBox, QComboBox,
                            QPushButton)
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

RADIUS = 100.
RATIO = .65
OFFSET = np.deg2rad(28.9)
CELLS = 6
WIDTH = RADIUS*np.sqrt(1+RATIO**2-2*RATIO*np.cos(OFFSET))
LENGTH = RADIUS*np.sqrt(1+RATIO**2)
COLORS = ('#16889b', '#d07838', '#8c62ad')
BACK_COLORS = ('#fff2ce', '#fff2ce', '#fff2ce')


class FabricFaces(Poly3DCollection):
    """Color the visible material side and relight it on every camera redraw."""
    def __init__(self, triangles, strap, opacity):
        self.triangles = np.asarray(triangles)
        self.front = np.array(to_rgb(COLORS[strap]))
        self.back = np.array(to_rgb(BACK_COLORS[strap]))
        super().__init__(triangles, alpha=opacity, edgecolor='none')

    def do_3d_projection(self):
        ax = self.axes
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        points = (self.triangles-limits[:, 0])*ax._box_aspect/np.diff(limits).ravel()
        normals = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        eye = ax._box_aspect/2 + ax._dist*ax._view_w
        front = np.sum(normals*(eye-points.mean(axis=1)), axis=1) >= 0
        visible = normals*np.where(front, 1., -1.)[:, None]
        light = ax._view_w + .6*ax._view_v - .5*ax._view_u
        light /= np.linalg.norm(light)
        diffuse = np.maximum(0, visible@light)
        brightness = np.where(front, .48 + .52*diffuse, .78 + .22*diffuse)
        colors = np.where(front[:, None], self.front, self.back)*brightness[:, None]
        self.set_facecolor(colors)
        return super().do_3d_projection()


def vertices(angle, cell, strap=0):
    """Material corners: lower left/right, upper left/right, in millimeters."""
    h = RADIUS*np.sqrt(2*RATIO*np.cos(np.deg2rad(angle)))
    phase = cell*np.deg2rad(angle)+strap*2*np.pi/3
    radii = np.array([1, RATIO, RATIO, 1] if cell%2 == 0 else [RATIO, 1, 1, RATIO])*RADIUS
    azimuth = phase+np.array([0, OFFSET, np.deg2rad(angle), np.deg2rad(angle)+OFFSET])
    return np.column_stack((radii*np.cos(azimuth), radii*np.sin(azimuth),
                            h*np.array([cell, cell, cell+1, cell+1])))


def cloth(v, x, y):
    a,b,c,d = v
    return a+x*(b-a)+y*(d-b) if y <= x else a+x*(d-c)+y*(c-a)


class FabricWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fabric assembly — 6 cells · r/R 0.65 · offset 28.9°')
        self.resize(1450, 950)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        side = QVBoxLayout()
        layout.addLayout(side)
        title = QLabel('6 cells · 3 continuous straps\nr/R = 0.65 · anchor offset 28.9°\nR = 100 mm')
        side.addWidget(title)
        self.angle_label = QLabel()
        side.addWidget(self.angle_label)
        self.angle = QSlider(Qt.Orientation.Horizontal)
        self.angle.setRange(0, 890)
        self.angle.setValue(450)
        side.addWidget(self.angle)
        self.cell = QComboBox()
        self.cell.addItems(['All six cells']+[f'Cell {j+1}' for j in range(CELLS)])
        side.addWidget(self.cell)
        self.strap = QComboBox()
        self.strap.addItems(['All three straps', 'Strap 1 — teal', 'Strap 2 — orange', 'Strap 3 — purple'])
        side.addWidget(self.strap)
        self.grid = QCheckBox('Show warp and weft')
        self.grid.setChecked(True)
        self.plates = QCheckBox('Show plate envelopes')
        self.plates.setChecked(True)
        self.labels = QCheckBox('Show corner labels')
        for box in (self.grid, self.plates, self.labels):
            side.addWidget(box)
            box.toggled.connect(self.draw)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(15, 100)
        self.opacity.setValue(95)
        side.addWidget(QLabel('Fabric opacity'))
        side.addWidget(self.opacity)
        self.play = QPushButton('Animate twist')
        side.addWidget(self.play)
        self.timer = QTimer(self)
        self.timer.setInterval(70)
        self.direction = 1
        self.timer.timeout.connect(self.step)
        self.play.clicked.connect(self.animate)
        for label, elev, azim in [('Assembly view', 18, -65), ('Top view', 90, -90), ('Side view', 0, -65)]:
            button = QPushButton(label)
            button.clicked.connect(lambda _, e=elev, a=azim: self.camera(e,a))
            side.addWidget(button)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setMaximumWidth(290)
        side.addWidget(self.info)
        note = QLabel('Front: saturated color.\nBack: pale cream.\nLighting follows the view.\n\nDashed black: flexible diagonal fold.\nSolid colored: long edges.\nGray bands: plate clamps.\n\nDo not cut or stitch the diagonals.\nUse slots or open bars at the plates.\n\nDrag the 3D view to rotate.\nThis shows an ideal cloth path;\nmaterial locking and stability at this\nnew offset are not validated.')
        note.setWordWrap(True)
        note.setMaximumWidth(290)
        side.addWidget(note)
        side.addStretch()
        main = QVBoxLayout()
        layout.addLayout(main, 1)
        self.figure = Figure(figsize=(11,8), facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.figure)
        main.addWidget(NavigationToolbar2QT(self.canvas, self))
        main.addWidget(self.canvas, 1)
        self.ax = self.figure.add_axes((.02,.07,.70,.88), projection='3d')
        self.pattern = self.figure.add_axes((.77,.12,.20,.76))
        self.ax.view_init(elev=18, azim=-65)
        for control in (self.angle, self.opacity):
            control.valueChanged.connect(self.draw)
        for control in (self.cell, self.strap):
            control.currentIndexChanged.connect(self.draw)
        self.draw()

    def camera(self, elev, azim):
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def animate(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText('Animate twist')
        else:
            self.timer.start()
            self.play.setText('Pause')

    def step(self):
        value = self.angle.value()+self.direction*5
        if value >= 890 or value <= 0:
            self.direction *= -1
        self.angle.setValue(max(0,min(890,value)))

    def draw(self):
        angle = self.angle.value()/10
        h = RADIUS*np.sqrt(2*RATIO*np.cos(np.deg2rad(angle)))
        self.angle_label.setText(f'Twist per cell: {angle:.1f}°')
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.clear()
        self.ax.view_init(elev=elev, azim=azim)
        cells = range(CELLS) if self.cell.currentIndex()==0 else [self.cell.currentIndex()-1]
        straps = range(3) if self.strap.currentIndex()==0 else [self.strap.currentIndex()-1]
        low, high = min(cells), max(cells)+1
        shear = []
        for j in cells:
            for k in straps:
                v = vertices(angle,j,k)
                self.ax.add_collection3d(FabricFaces(v[[[0,1,3],[0,3,2]]],
                    k, self.opacity.value()/100))
                for ids in ([0,2],[1,3]):
                    self.ax.plot(*v[ids].T, color=COLORS[k], lw=3)
                for ids in ([0,1],[2,3]):
                    self.ax.plot(*v[ids].T, color='#555555', lw=3)
                self.ax.plot(*v[[0,3]].T, color='#202020', lw=1.5, linestyle='--')
                c = np.dot((v[1]-v[0])/WIDTH,(v[3]-v[1])/LENGTH)
                shear.append(abs(np.rad2deg(np.arcsin(np.clip(c,-1,1)))))
                if self.grid.isChecked():
                    for u in np.linspace(0,1,9)[1:-1]:
                        for points in ([cloth(v,u,t) for t in (0,u,1)], [cloth(v,t,u) for t in (0,u,1)]):
                            self.ax.plot(*np.array(points).T,color=COLORS[k],lw=.6,alpha=.9)
                if self.labels.isChecked():
                    for name,p in zip(('A','B','C','D'),v):
                        self.ax.text(*p,f'{j+1}{name}',fontsize=8)
        if self.plates.isChecked():
            t = np.linspace(0,2*np.pi,90)
            for j in range(low,high+1):
                rim = np.column_stack((RADIUS*np.cos(t),RADIUS*np.sin(t),np.full(len(t),j*h)))
                self.ax.plot(*rim.T,color='#89939a',lw=.8)
                self.ax.text(RADIUS*1.1,0,j*h,f'Plate {j}',fontsize=9)
        self.ax.set(xlim=(-120,120),ylim=(-120,120),zlim=(low*h-5,high*h+5))
        self.ax.set_box_aspect((240,240,(high-low)*h+10))
        self.ax.set_axis_off()
        self.ax.set_title('Assembly geometry · dimensions in mm')
        self.pattern.clear()
        pitch = LENGTH+10
        for j in range(CELLS+1):
            y=j*pitch
            self.pattern.fill_between([0,WIDTH],y,y+10,color='#cccccc')
            self.pattern.plot([0,WIDTH],[y+5,y+5],color='#666',lw=.6,linestyle=':')
            self.pattern.text(WIDTH+3,y+5,str(j),fontsize=8,va='center')
            if j<CELLS:
                color='#202020' if j in cells else '#bcbcbc'
                self.pattern.plot([0,WIDTH],[y+10,y+10+LENGTH],color=color,linestyle='--',lw=1)
        total=CELLS*LENGTH+(CELLS+1)*10
        self.pattern.plot([0,0,WIDTH,WIDTH,0],[0,total,total,0,0],color='#555',lw=1)
        self.pattern.text(0,13,'A',fontsize=8)
        self.pattern.text(WIDTH,13,'B',fontsize=8,ha='right')
        self.pattern.text(0,pitch-3,'C',fontsize=8)
        self.pattern.text(WIDTH,pitch-3,'D',fontsize=8,ha='right')
        self.pattern.set(xlim=(-7,WIDTH+15),ylim=(total+10,-10))
        self.pattern.set_aspect('equal')
        self.pattern.set_axis_off()
        self.pattern.set_title('Flat strap\nA–D folds',fontsize=10)
        self.info.setText(f'Cell height: {h:.2f} mm\nStack height: {CELLS*h:.2f} mm\nActive width: {WIDTH:.2f} mm\nFree cell length: {LENGTH:.2f} mm\nLargest shown weave shear: {max(shear):.1f}°\n\nFlat pattern excludes hems.\n10 mm clamp bands fold in half.')
        self.canvas.draw_idle()


def main():
    app=QApplication(sys.argv)
    window=FabricWindow()
    window.show()
    sys.exit(app.exec())


if __name__=='__main__':
    main()
