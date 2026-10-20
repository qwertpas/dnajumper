"""Rectangular woven straps: cut outline, hems, clamp bands, and fold marks.

All exported dimensions are millimeters. Diagonals are marks, never cuts or
inextensible seams. The fabric must permit at least 35.2 degrees of trellis
shear for the default r/R=.8 design; this is not a paper-origami pattern.
"""

from pathlib import Path
from math import cos, pi, sqrt

import ezdxf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).parent
RADIUS, RATIO, BAND, HEM = 100., .8, 10., 5.
WIDTH = RADIUS*sqrt(1+RATIO**2-RATIO)
LENGTH = RADIUS*sqrt(1+RATIO**2)
COLORS = ('#147d92', '#b85c32', '#805ca3')


def lines(cells):
    """Return (layer, x1,y1,x2,y2) with physical cut coordinates."""
    width, height = WIDTH+2*HEM, cells*LENGTH+(cells+1)*BAND
    result = [('CUT_OUTLINE', 0, 0, width, 0),
              ('CUT_OUTLINE', width, 0, width, height),
              ('CUT_OUTLINE', width, height, 0, height),
              ('CUT_OUTLINE', 0, height, 0, 0)]
    for x in (HEM, HEM+WIDTH):
        result.append(('MARK_HEM', x, 0, x, height))
    for j in range(cells+1):
        start = j*(LENGTH+BAND)
        for y in (start, start+BAND):
            result.append(('MARK_CLAMP', HEM, y, HEM+WIDTH, y))
        result.append(('MARK_FOLD', HEM, start+BAND/2, HEM+WIDTH, start+BAND/2))
        if j < cells:
            result.append(('MARK_DIAGONAL', HEM, start+BAND,
                           HEM+WIDTH, start+BAND+LENGTH))
    return result, width, height


def svg(cells):
    marks, width, height = lines(cells)
    margin = 14
    result = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width+2*margin:.6f}mm" '
              f'height="{height+2*margin:.6f}mm" viewBox="{-margin} {-margin} {width+2*margin} {height+2*margin}">',
              '<title>Woven strap cutting and marking template in millimeters</title>',
              '<desc>Cut only the outline. Fold five millimeter hems, hairpin fold ten millimeter clamp bands. '
              'Mark diagonals without cutting, stitching or taping them.</desc>']
    for j in range(cells+1):
        y = j*(LENGTH+BAND)
        result.append(f'<rect x="{HEM}" y="{y}" width="{WIDTH}" height="{BAND}" fill="#e6e6e6"/>')
        result.append(f'<text x="{width+2}" y="{y+6}" font-size="3" font-family="sans-serif">{j}</text>')
    for layer,x1,y1,x2,y2 in marks:
        color = '#151515' if layer == 'CUT_OUTLINE' else '#186b9b'
        dash = '' if layer == 'CUT_OUTLINE' else ' stroke-dasharray="2 1"'
        result.append(f'<line data-layer="{layer}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                      f'stroke="{color}" stroke-width="0.25"{dash}/>')
    result += [f'<text x="0" y="-7" font-size="3.3" font-family="sans-serif">'
               f'{cells} cells | cut {width:.3f} × {height:.3f} mm</text>',
               '<text x="0" y="-2" font-size="2.7" font-family="sans-serif">'
               'Black: cut outline · Blue: mark/fold only · Gray: clamp</text>', '</svg>']
    (ROOT/f'strap_{cells}_cells.svg').write_text('\n'.join(result)+'\n')


def dxf(cells):
    marks, _, _ = lines(cells)
    doc = ezdxf.new('R2013', units=ezdxf.units.MM)
    for layer,color in (('CUT_OUTLINE',7),('MARK_HEM',5),('MARK_CLAMP',8),
                        ('MARK_FOLD',3),('MARK_DIAGONAL',4)):
        doc.layers.new(layer, dxfattribs={'color':color})
    drawing = doc.modelspace()
    for layer,x1,y1,x2,y2 in marks:
        drawing.add_line((x1,y1), (x2,y2), dxfattribs={'layer':layer})
    doc.saveas(ROOT/f'strap_{cells}_cells.dxf')


def cell(theta, parity, phase=0., z=0.):
    r = RATIO
    h = sqrt(max(0., 2*r*cos(theta)))
    radii = np.array([1,r,r,1]) if parity == 0 else np.array([r,1,1,r])
    angles = np.array([phase, phase+pi/3, phase+theta, phase+pi/3+theta])
    return np.column_stack((radii*np.cos(angles), radii*np.sin(angles),
                            z+np.array([0,0,h,h])))


def cloth(vertices, x, y):
    """Map normalized material coordinates by a crease along x=y."""
    a,b,c,d = vertices
    return (a+x*(b-a)+y*(d-b)) if y <= x else (a+x*(d-c)+y*(c-a))


def draw_stack(ax, degrees, cells=3):
    theta = np.deg2rad(degrees)
    h = sqrt(max(0., 2*RATIO*cos(theta)))
    for j in range(cells+1):
        t = np.linspace(0,2*pi,100)
        rim = np.column_stack((np.cos(t),np.sin(t),np.full(len(t), j*h)))
        ax.add_collection3d(Poly3DCollection([rim],facecolor='#c8cdd1',edgecolor='#909da4',
                                           linewidth=.5,alpha=.18))
    for j in range(cells):
        for k in range(3):
            v = cell(theta,j%2,j*theta+k*2*pi/3,j*h)
            triangles = v[np.array([[0,1,3],[0,3,2]])]
            ax.add_collection3d(Poly3DCollection(triangles,facecolor=COLORS[k],alpha=.35,
                                               edgecolor='none'))
            for u in np.linspace(0,1,9):
                warp = np.array([cloth(v,u,t) for t in (0,u,1)])
                weft = np.array([cloth(v,t,u) for t in (0,u,1)])
                ax.plot(*warp.T,color=COLORS[k],lw=.65)
                ax.plot(*weft.T,color=COLORS[k],lw=.45,alpha=.7)
            ax.plot(*v[[0,2]].T,color=COLORS[k],lw=2.5)
            ax.plot(*v[[1,3]].T,color=COLORS[k],lw=2.5)
            ax.plot(*v[[0,3]].T,color='#343a40',lw=.7,linestyle='--')
    ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-.05,max(.1,cells*h+.05)))
    ax.set_box_aspect((2.2,2.2,max(.4,cells*h)))
    ax.view_init(elev=20,azim=-68)
    ax.set_axis_off()


def figure():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig = plt.figure(figsize=(14,7),layout='constrained')
    grid=fig.add_gridspec(1,4,width_ratios=(.95,1.3,1,1))
    ax=fig.add_subplot(grid[0,0])
    marks,width,height=lines(3)
    for j in range(4):
        y=j*(LENGTH+BAND)
        ax.add_patch(plt.Rectangle((HEM,y),WIDTH,BAND,color='#d9d9d9'))
        ax.text(width+4,y+BAND/2,f'Plate {j}',va='center',fontsize=9)
    for layer,x1,y1,x2,y2 in marks:
        ax.plot([x1,x2],[y1,y2],color='#222222' if layer=='CUT_OUTLINE' else '#167496',
                lw=1 if layer=='CUT_OUTLINE' else .7,
                linestyle='-' if layer=='CUT_OUTLINE' else '--')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlim(-10,width+43)
    ax.axis('off')
    ax.set_title('One continuous rectangle\nNo internal cuts')
    ax.text(width/2,height+22,f'{width:.2f} mm raw width\n5 mm hems · 10 mm clamp bands',
            ha='center',va='top',fontsize=9)
    for col,degree,count in ((1,45,3),(2,15,1),(3,75,1)):
        ax=fig.add_subplot(grid[0,col],projection='3d')
        draw_stack(ax,degree,count)
        ax.set_title(f'{count} cell'+('s' if count>1 else '')+f' · {degree}° twist/cell')
    fig.savefig(ROOT/'woven_straps.png',dpi=200)
    plt.close(fig)


def plate_svg():
    width, height = 4.5*RADIUS, 2.5*RADIUS
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">',
         '<title>Alternating plate clamp layouts in millimeters</title>']
    for parity,xc in enumerate((1.12*RADIUS,3.37*RADIUS)):
        yc=1.2*RADIUS
        out.append(f'<circle cx="{xc}" cy="{yc}" r="{RADIUS}" fill="none" stroke="#555" stroke-width=".4"/>')
        out.append(f'<text x="{xc}" y="{.1*RADIUS}" text-anchor="middle" font-family="sans-serif" font-size="5">Plate {"A" if parity==0 else "B"}</text>')
        for k in range(3):
            rr=(RADIUS,RADIUS*RATIO) if parity==0 else (RADIUS*RATIO,RADIUS)
            ph=k*2*pi/3
            a=rr[0]*np.array([np.cos(ph),np.sin(ph)])
            b=rr[1]*np.array([np.cos(ph+pi/3),np.sin(ph+pi/3)])
            tangent=(b-a)/WIDTH
            normal=np.array([-tangent[1],tangent[0]])
            if normal@(a+b)>0:normal=-normal
            corners=np.array([a,b,b+BAND/2*normal,a+BAND/2*normal])+[xc,yc]
            xy=' '.join(f'{x:.6f},{y:.6f}' for x,y in corners)
            out.append(f'<polygon points="{xy}" fill="#ddd" stroke="{COLORS[k]}" stroke-width=".35"/>')
            for name,p in (('Left',a),('Right',b)):
                x,y=p+[xc,yc]
                out.append(f'<circle cx="{x}" cy="{y}" r="1" fill="{COLORS[k]}"/>')
                label=p*.9+[xc,yc]
                out.append(f'<text x="{label[0]}" y="{label[1]}" text-anchor="middle" font-family="sans-serif" font-size="3">{k+1}{name[0]}</text>')
    out.append(f'<text x="{width/2}" y="{2.34*RADIUS}" text-anchor="middle" font-family="sans-serif" font-size="4">'
               f'R={RADIUS:g} mm · r={RADIUS*RATIO:g} mm · seam {WIDTH:.3f} mm · folded band depth {BAND/2:g} mm</text>')
    out.append(f'<text x="{width/2}" y="{2.43*RADIUS}" text-anchor="middle" font-family="sans-serif" font-size="3.5">'
               'Circles: envelopes only. Provide slots or open clamp bars along the seams.</text></svg>')
    (ROOT/'plate_clamps.svg').write_text('\n'.join(out)+'\n')


if __name__=='__main__':
    for cells in (3,12):
        svg(cells)
        dxf(cells)
    plate_svg()
    figure()
    print(f'Active width={WIDTH:.6f} mm, active pitch={LENGTH:.6f} mm')
    print(f'12 cells: raw width={WIDTH+2*HEM:.6f}, length={12*LENGTH+13*BAND:.6f} mm')
