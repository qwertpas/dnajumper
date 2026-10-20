"""Three continuous, shear-free routed strips: python3.11 -m trussbuckling.research.no_shear.gui."""
import sys
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                            QVBoxLayout, QLabel, QSlider, QComboBox, QPushButton)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from ...fabric_gui import FabricFaces, COLORS
from .routed import strip_mesh, RADIUS, RATIO, CELLS, LENGTH, WIDTH, FAN_RADIUS


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Three routed fabric strips — no weave shear')
        self.resize(1500,950)
        root=QWidget()
        self.setCentralWidget(root)
        layout=QHBoxLayout(root)
        side=QVBoxLayout()
        layout.addLayout(side)
        side.addWidget(QLabel('6 cells · 3 continuous textile pieces\n6 independent tension paths\nr/R = 0.65 · anchor offset 28.9°'))
        self.readout=QLabel()
        side.addWidget(self.readout)
        self.angle=QSlider(Qt.Orientation.Horizontal)
        self.angle.setRange(110,650)
        self.angle.setValue(450)
        side.addWidget(self.angle)
        self.strip=QComboBox()
        self.strip.addItems(['All three strips','Strip 1','Strip 2','Strip 3'])
        side.addWidget(self.strip)
        self.joint=QComboBox()
        self.joint.addItems(['Complete stack']+[f'Fold detail {i}' for i in range(1,13)])
        side.addWidget(self.joint)
        for label,elev,azim in [('Assembly view',18,-65),('Top view',90,-90),('Side view',0,-65)]:
            button=QPushButton(label)
            button.clicked.connect(lambda _,e=elev,a=azim:self.camera(e,a))
            side.addWidget(button)
        note=QLabel('Solid center lines carry tension.\nDo not fold the load diamonds.\nDashed lines: unloaded fan folds.\nDots: ideal swivel centers.\n\nEach piece runs up one side,\nacross the top plate, then down\nthe other side. Clamp every span\nindependently: no sliding through.\n\nFront: saturated color.\nBack: cream. Drag to rotate.\n\nThis is an ideal geometry prototype.\nJoint clearance, grips and strength\nrequire hardware design.\nShown range stops before the\ncurrent fan branch becomes singular.')
        note.setWordWrap(True)
        note.setMaximumWidth(285)
        side.addWidget(note)
        side.addStretch()
        body=QVBoxLayout()
        layout.addLayout(body,1)
        self.figure=Figure(figsize=(12,8),facecolor='white')
        self.canvas=FigureCanvasQTAgg(self.figure)
        body.addWidget(NavigationToolbar2QT(self.canvas,self))
        body.addWidget(self.canvas)
        self.ax=self.figure.add_axes((.01,.06,.67,.89),projection='3d')
        self.flat=self.figure.add_axes((.73,.10,.25,.80))
        self.ax.view_init(elev=18,azim=-65)
        self.angle.valueChanged.connect(self.draw)
        self.strip.currentIndexChanged.connect(self.draw)
        self.joint.currentIndexChanged.connect(self.draw)
        self.draw()

    def camera(self,elev,azim):
        self.ax.view_init(elev=elev,azim=azim)
        self.canvas.draw_idle()

    def draw(self):
        degree=self.angle.value()/10
        theta=np.deg2rad(degree)
        joint=self.joint.currentIndex()
        chosen=max(0,self.strip.currentIndex()-1)
        strips=range(3) if self.strip.currentIndex()==0 and not joint else [chosen]
        height=RADIUS*np.sqrt(2*RATIO*np.cos(theta))
        self.readout.setText(f'Twist / cell: {degree:.1f}°\nCell height: {height:.2f} mm\nLoad diagonal: {LENGTH:.2f} mm\nDiamond width: {WIDTH:g} mm')
        elev,azim=self.ax.elev,self.ax.azim
        self.ax.clear()
        self.ax.view_init(elev=elev,azim=azim)
        selected=strip_mesh(theta,chosen)
        for k in strips:
            model=strip_mesh(theta,k)
            faces=[]
            for record in model['triangles']:
                if joint and (record['kind']=='fan' and record['index']!=joint or
                              record['kind']!='fan' and record['index'] not in (joint-1,joint)):
                    continue
                faces.append(record['world'])
                if record['kind']=='fan':
                    p=np.vstack((record['world'],record['world'][0]))
                    self.ax.plot(*p.T,color='#333333',lw=.7,linestyle='--')
            self.ax.add_collection3d(FabricFaces(faces,k,1.))
            nodes=model['nodes']
            for j,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
                if joint and j not in (joint-1,joint):continue
                self.ax.plot(*np.array([a,b]).T,color='#303030',lw=1.2)
            self.ax.scatter(*nodes.T,s=10,color=COLORS[k])
        if joint:
            center=selected['nodes'][joint]
            radius=FAN_RADIUS*3
            self.ax.set(xlim=(center[0]-radius,center[0]+radius),
                        ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
            self.ax.set_box_aspect((1,1,1))
            self.ax.set_title(f'Swivel {joint}: three unloaded fold axes')
        else:
            t=np.linspace(0,2*np.pi,100)
            for j in range(CELLS+1):
                self.ax.plot(RADIUS*np.cos(t),RADIUS*np.sin(t),np.full(len(t),j*height),color='#999999',lw=.7)
            self.ax.set(xlim=(-120,120),ylim=(-120,120),zlim=(-5,CELLS*height+5))
            self.ax.set_box_aspect((240,240,CELLS*height+10))
            self.ax.set_title('Flat load spans; folding is confined to the joints')
        self.ax.set_axis_off()
        self.flat.clear()
        for record in selected['triangles']:
            p=record['flat'][:,:2]
            self.flat.fill(p[:,0],p[:,1],color=COLORS[chosen],alpha=.2 if record['kind']=='fan' else .6)
            if record['kind']=='fan':
                p=np.vstack((p,p[0]))
                self.flat.plot(p[:,0],p[:,1],color='#333',lw=.6,linestyle='--')
        if joint:
            x,y=selected['net']['nodes'][joint,:2]
            r=FAN_RADIUS*3
            self.flat.set(xlim=(x-r,x+r),ylim=(y-r,y+r))
            self.flat.set_title('Same joint, unfolded')
        else:
            self.flat.set_title('One continuous cut piece\nLoad paths follow warp / weft',fontsize=10)
        self.flat.set_aspect('equal')
        self.flat.set_axis_off()
        self.canvas.draw_idle()


def main():
    app=QApplication(sys.argv)
    window=Window()
    window.show()
    sys.exit(app.exec())


if __name__=='__main__':main()
