"""Generate a one-piece twelve-cell textile blank and a repeated plate pattern.

Run from the repo root: python3.11 -m trussbuckling.make_templates
Lengths are millimeters. These are prototype geometry/bonding templates;
no fabric, adhesive or plate load rating is inferred from the cord model.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from html import escape
import json
from math import ceil, cos, pi, sin, sqrt
from pathlib import Path
import zipfile

import ezdxf
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


@dataclass
class Dimensions:
    R: float = 100.
    ratio: float = .65
    cells: int = 12
    rail: float = 6.
    grip_width: float = 14.
    grip: float = 20.
    plate_thickness: float = 3.
    fabric_thickness: float = .25
    plate_margin: float = 15.
    slot_depth: float = 4.
    slot_clearance: float = 1.
    fold_slack: float = 1.
    carrier_width: float = 2.5
    carrier_gap: float = 8.
    carrier_slack: float = 15.


def build(d):
    if not 2-sqrt(3) < d.ratio < 1 or d.R <= 0 or d.cells < 1:
        raise ValueError("Use R > 0, cells >= 1 and 0.268 < r/R < 1 for the 90-degree design.")
    if not 0 < d.rail <= d.grip_width or d.grip <= d.plate_thickness+d.fold_slack:
        raise ValueError("Grip must be wider than the rail and long enough for the plate route.")
    if d.carrier_gap <= 2*d.carrier_width:
        raise ValueError("The carrier gap must exceed twice the carrier width.")
    r = d.R*d.ratio
    length = sqrt(d.R**2+r**2)
    height = sqrt(2*d.R*r)
    root_spacing = sqrt(d.R**2+r**2-d.R*r)
    land = (d.grip-d.plate_thickness-d.fold_slack)/2
    slot_offset = land+d.slot_depth/2
    if d.plate_margin <= land+d.slot_depth+2:
        raise ValueError("Increase plate margin: slots need at least 2 mm of remaining outer rim.")
    pitch = d.grip_width+d.carrier_gap
    # The joining loops are deliberately longer than their installed endpoint
    # separation even when the endpoints are routed outward onto grip tabs.
    a, b = d.R+slot_offset, r+slot_offset
    carrier_distance = sqrt(a*a+b*b-a*b)+d.grip_width
    # The loop can touch a widened grip before its midpoint. Count only the
    # free path above the FIRST edge of that grip, a conservative lower bound.
    rise = 5*ceil((carrier_distance+d.carrier_slack-d.carrier_gap
                   +2*d.carrier_width+d.grip)/10)
    carrier_length = 2*(rise-d.grip/2)+d.carrier_gap-2*d.carrier_width
    first = rise-d.grip/2
    total = first+d.cells*(length+d.grip)+d.grip
    width = 5*pitch+d.grip_width
    centers = [d.grip_width/2+i*pitch for i in range(6)]
    rows = [{"plate": j, "y0": first+j*(length+d.grip),
             "y1": first+j*(length+d.grip)+d.grip, "clock": (60*j) % 360,
             "lane_to_root": [(i-j) % 6+1 for i in range(6)]} for j in range(d.cells+1)]
    pieces = []
    lines, texts = [], []
    for i, x in enumerate(centers):
        pieces.append(box(x-d.rail/2, first, x+d.rail/2, total))
        for row in rows:
            y0, y1 = row["y0"], row["y1"]
            taper = min(3., d.grip/4)
            pieces.append(Polygon([(x-d.rail/2, y0), (x+d.rail/2, y0),
                (x+d.grip_width/2, y0+taper), (x+d.grip_width/2, y1-taper),
                (x+d.rail/2, y1), (x-d.rail/2, y1),
                (x-d.grip_width/2, y1-taper), (x-d.grip_width/2, y0+taper)]))
            for y in (y0, y1):
                lines.append({"points": [[x-d.rail/2, y], [x+d.rail/2, y]],
                              "color": "#1676a7", "width": .18})
            texts += [{"x": x, "y": y0+6, "text": f"P{row['plate']:02d}",
                       "size": 2., "anchor": "middle", "angle": 0, "color": "#333333"},
                      {"x": x, "y": y0+14, "text": f"L{i+1}>K{row['lane_to_root'][i]}",
                       "size": 1.7, "anchor": "middle", "angle": 0, "color": "#333333"}]
        for j in range(d.cells):
            texts.append({"x": x, "y": rows[j]["y1"]+length/2,
                "text": f"L{i+1}", "size": 1.6, "anchor": "middle", "angle": 0, "color": "#666666"})
    # Five U carriers join the first grip pads only. All other fingers are free.
    for i in range(5):
        left = centers[i]+d.grip_width/2
        right = centers[i+1]-d.grip_width/2
        pieces += [box(left-.3, 0, left+d.carrier_width, rise+1),
                   box(right-d.carrier_width, 0, right+.3, rise+1),
                   box(left, 0, right, d.carrier_width)]
    fabric_shape = unary_union(pieces)
    if fabric_shape.geom_type != "Polygon" or not fabric_shape.is_valid or fabric_shape.interiors:
        raise ValueError("The fabric profile must be one connected piece without loose cut islands.")
    fabric = {"name": "Fabric blank", "quantity": 1, "width": width, "height": total,
              "cut": [list(map(list, fabric_shape.exterior.coords))], "lines": lines, "text": texts}

    outer = d.R+d.plate_margin
    angles = np.linspace(0, 2*pi, 721)
    circle = np.column_stack((outer+outer*np.cos(angles), outer+outer*np.sin(angles))).tolist()
    plate = {"name": "Repeated plate", "quantity": d.cells+1, "width": 2*outer,
             "height": 2*outer, "circle": [outer, outer, outer], "cut": [circle],
             "lines": [], "text": []}
    roots, holes = [], []
    slot_width = d.grip_width+d.slot_clearance
    for k in range(6):
        angle = k*pi/3
        radius = d.R if k % 2 == 0 else r
        axis = np.array([cos(angle), sin(angle)])
        tangent = np.array([-sin(angle), cos(angle)])
        root = np.array([outer, outer])+radius*axis
        # Slot near edge: radial inner edge = root radius + land, not its center.
        corner = min(.8, d.slot_depth/4)
        hole = box(-d.slot_depth/2+corner, -slot_width/2+corner,
                   d.slot_depth/2-corner, slot_width/2-corner).buffer(corner, quad_segs=8)
        hole = rotate(hole, 60*k, origin=(0, 0))
        slot_center = np.array([outer, outer])+(radius+slot_offset)*axis
        hole = translate(hole, *slot_center)
        holes.append(hole)
        plate["cut"].append(list(map(list, hole.exterior.coords)))
        for direction in (axis, tangent):
            plate["lines"].append({"points": [(root-2*direction).tolist(), (root+2*direction).tolist()],
                                    "color": "#1676a7", "width": .15})
        pad = [root-d.grip_width/2*tangent, root+d.grip_width/2*tangent,
               root+land*axis+d.grip_width/2*tangent, root+land*axis-d.grip_width/2*tangent]
        plate["lines"].append({"points": [p.tolist() for p in pad]+[pad[0].tolist()],
                               "color": "#7a858e", "width": .12, "dash": [1, 1]})
        label = np.array([outer, outer])+(radius-7)*axis
        plate["text"].append({"x": label[0], "y": label[1], "text": f"K{k+1}",
            "size": 2.8, "anchor": "middle", "angle": 0, "color": "#333333"})
        roots.append({"root": k+1, "radius": radius, "angle": 60*k,
                      "x": root[0], "y": root[1], "slot_center": slot_center.tolist()})
    plate["lines"] += [{"points": [[outer-4, outer], [outer+4, outer]], "color": "#1676a7", "width": .12},
                       {"points": [[outer, outer-4], [outer, outer+4]], "color": "#1676a7", "width": .12}]
    plate["text"] += [{"x": outer, "y": outer-17, "text": "R100 / r65" if d.R == 100 and r == 65 else f"R{d.R:g} / r{r:g}",
        "size": 3, "anchor": "middle", "angle": 0, "color": "#333333"},
        {"x": outer, "y": outer+17, "text": "K1 is clock zero", "size": 2.5,
         "anchor": "middle", "angle": 0, "color": "#333333"}]
    plate_shape = Polygon(circle, [list(h.exterior.coords) for h in holes])
    if not plate_shape.is_valid or any(h.distance(Polygon(circle).boundary) < 2 for h in holes):
        raise ValueError("Invalid plate slots or insufficient remaining rim.")
    params = asdict(d) | {"r": r, "L": length, "H0": height, "root_spacing": root_spacing,
        "plate_radius": outer, "land": land, "slot_width": slot_width,
        "slot_offset": slot_offset, "lane_pitch": pitch, "carrier_rise": rise,
        "carrier_length": carrier_length, "carrier_endpoint_bound": carrier_distance,
        "initial_center_pitch": height+d.plate_thickness,
        "initial_total_height": d.cells*height+(d.cells+1)*d.plate_thickness}
    return {"units": "mm", "parameters": params, "fabric": fabric, "plate": plate,
        "rows": rows, "root_sites": roots, "verification": {
            "fabric_components": 1, "fabric_holes": 0, "fabric_area_mm2": fabric_shape.area,
            "free_spans": d.cells*6, "grips": (d.cells+1)*6,
            "carrier_slack_lower_bound_mm": carrier_length-carrier_distance,
            "minimum_plate_rim_mm": min(h.distance(Polygon(circle).boundary) for h in holes)}}


def save_svg(sheet, path, marked):
    text = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{sheet["width"]:.6f}mm" '
            f'height="{sheet["height"]:.6f}mm" viewBox="0 0 {sheet["width"]:.6f} {sheet["height"]:.6f}">',
            '<g id="CUT" fill="none" stroke="black" stroke-width="0.1">']
    for points in sheet["cut"]:
        data = " ".join(f'{x:.6f},{y:.6f}' for x, y in points)
        text.append(f'<polygon points="{data}"/>')
    text.append('</g>')
    if marked:
        text.append('<g id="MARK">')
        for line in sheet["lines"]:
            data = " ".join(f'{x:.6f},{y:.6f}' for x, y in line["points"])
            dash = ' stroke-dasharray="1 1"' if line.get("dash") else ''
            text.append(f'<polyline points="{data}" fill="none" stroke="{line["color"]}" '
                        f'stroke-width="{line["width"]}"{dash}/>')
        for item in sheet["text"]:
            text.append(f'<text x="{item["x"]:.6f}" y="{item["y"]:.6f}" font-family="sans-serif" '
                f'font-size="{item["size"]}" text-anchor="{item["anchor"]}" fill="{item["color"]}">'
                f'{escape(item["text"])}</text>')
        text.append('</g>')
    text.append('</svg>')
    path.write_text("\n".join(text))


def save_dxf(sheet, path):
    doc = ezdxf.new("R2010")
    doc.units = 4
    doc.header["$MEASUREMENT"] = 1
    doc.layers.new("CUT", dxfattribs={"color": 7})
    doc.layers.new("MARK", dxfattribs={"color": 5})
    space = doc.modelspace()
    start = 0
    if "circle" in sheet:
        x, y, radius = sheet["circle"]
        space.add_circle((x, -y), radius, dxfattribs={"layer": "CUT"})
        start = 1
    for points in sheet["cut"][start:]:
        space.add_lwpolyline([(x, -y) for x, y in points[:-1]], close=True, dxfattribs={"layer": "CUT"})
    for line in sheet["lines"]:
        space.add_lwpolyline([(x, -y) for x, y in line["points"]], dxfattribs={"layer": "MARK"})
    doc.saveas(path)


def back_marks(plate):
    """Mirror the placement stencil for a left-to-right flip of the plate."""
    back = deepcopy(plate)
    width = plate["width"]
    back["cut"] = [[[width-x, y] for x, y in points] for points in back["cut"]]
    for line in back["lines"]:
        line["points"] = [[width-x, y] for x, y in line["points"]]
    for item in back["text"]:
        item["x"] = width-item["x"]
        if item["text"] == "K1 is clock zero":
            item["text"] = "BACK - clock from front"
    return back


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=100.)
    parser.add_argument("--ratio", type=float, default=.65)
    parser.add_argument("--cells", type=int, default=12)
    parser.add_argument("--plate-thickness", type=float, default=3.)
    parser.add_argument("--fabric-thickness", type=float, default=.25)
    parser.add_argument("--output", type=Path, default=Path("trussbuckling/templates/default"))
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    data = build(Dimensions(R=args.radius, ratio=args.ratio, cells=args.cells,
        plate_thickness=args.plate_thickness, fabric_thickness=args.fabric_thickness))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("fabric", "plate"):
        save_svg(data[name], args.output/f"{name}_cut.svg", marked=False)
        save_svg(data[name], args.output/f"{name}_marked.svg", marked=True)
        save_dxf(data[name], args.output/f"{name}.dxf")
    save_svg(back_marks(data["plate"]), args.output/"plate_back_marked.svg", marked=True)
    (args.output/"layout.json").write_text(json.dumps(data, indent=2))
    if not args.no_pdf:
        from .template_pdf import create_pdfs
        pdfs = create_pdfs(args.output/"layout.json", Path("output/pdf"))
        bundle = Path("output/strap_templates.zip")
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in args.output.iterdir():
                if path.is_file():
                    archive.write(path, "cut_files/"+path.name)
            for path in pdfs:
                archive.write(path, "pdf/"+path.name)
            for name in ("make_templates.py", "template_pdf.py", "requirements-templates.txt"):
                path = Path(__file__).with_name(name)
                archive.write(path, "source/trussbuckling/"+name)
    print(json.dumps({"fabric_size_mm": [data["fabric"]["width"], data["fabric"]["height"]],
        "L_mm": data["parameters"]["L"], "verification": data["verification"]}, indent=2))


if __name__ == "__main__":
    main()
