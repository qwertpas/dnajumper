"""Print-scale cutting sheets and assembly notes from the shared template layout.

All layout coordinates are millimetres, measured from the upper-left corner.
The drawing data, rather than the PDF renderer, defines every cutting edge.
"""

from __future__ import annotations

import argparse
import json
from math import ceil, sqrt
from pathlib import Path

from reportlab.lib.colors import HexColor, black
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


BLUE = HexColor("#185a8d")
GRAY = HexColor("#666666")
PALE = HexColor("#edf2f5")
PAGE = (210, 297)


def clean(text):
    return str(text).replace("\u2013", "-").replace("\u2014", "-")


def new_pdf(path, size):
    pdf = canvas.Canvas(str(path), pagesize=(size[0] * mm, size[1] * mm),
                        pageCompression=1)
    pdf.setTitle(path.stem.replace("_", " "))
    pdf.setAuthor("dnajumper / trussbuckling")
    pdf.scale(mm, mm)
    return pdf


def next_page(pdf, size=PAGE):
    pdf.showPage()
    pdf.setPageSize((size[0] * mm, size[1] * mm))
    pdf.scale(mm, mm)


def label(pdf, x, y, text, size=3, color=black, bold=False, align="start"):
    pdf.setFillColor(color)
    pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    draw = {"start": pdf.drawString, "middle": pdf.drawCentredString,
            "end": pdf.drawRightString}[align]
    draw(x, y, clean(text))


def paragraph(pdf, x, y, text, width=178, size=3.2, leading=4.8, color=black):
    """A small word wrapper keeps the four instruction pages easy to edit."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    for block in text.split("\n"):
        line = ""
        for word in block.split():
            trial = (line + " " + word).strip()
            if line and stringWidth(clean(trial), "Helvetica", size) > width:
                label(pdf, x, y, line, size, color)
                y -= leading
                line = word
            else:
                line = trial
        label(pdf, x, y, line, size, color)
        y -= leading
    return y


def sheet(pdf, data, x, top, scale=1, marks=True):
    pdf.saveState()
    pdf.setStrokeColor(black)
    pdf.setLineWidth(max(.08, .14 * scale))
    for points in data["cut"]:
        path = pdf.beginPath()
        path.moveTo(x + points[0][0] * scale, top - points[0][1] * scale)
        for px, py in points[1:]:
            path.lineTo(x + px * scale, top - py * scale)
        path.close()
        pdf.drawPath(path, stroke=1, fill=0)
    if marks:
        for item in data.get("lines", []):
            pdf.setStrokeColor(HexColor(item.get("color", "#185a8d")))
            pdf.setLineWidth(max(.07, item.get("width", .12) * scale))
            pdf.setDash([v * scale for v in item.get("dash", [])])
            path = pdf.beginPath()
            path.moveTo(x + item["points"][0][0] * scale,
                        top - item["points"][0][1] * scale)
            for px, py in item["points"][1:]:
                path.lineTo(x + px * scale, top - py * scale)
            pdf.drawPath(path)
        pdf.setDash()
        for item in data.get("text", []):
            pdf.saveState()
            pdf.translate(x + item["x"] * scale, top - item["y"] * scale)
            pdf.rotate(-item.get("angle", 0))
            label(pdf, 0, 0, item["text"], item.get("size", 2) * scale,
                  HexColor(item.get("color", "#185a8d")),
                  align=item.get("anchor", "start"))
            pdf.restoreState()
    pdf.restoreState()


def calibration(pdf, x, y):
    pdf.setStrokeColor(black)
    pdf.setLineWidth(.2)
    pdf.line(x, y, x + 100, y)
    for pos in (0, 50, 100):
        pdf.line(x + pos, y - 1, x + pos, y + 1)
    label(pdf, x + 50, y + 2.4, "100 mm - verify with a ruler", 2.5, align="middle")


def full_size(data, path):
    fabric, plate, p = data["fabric"], data["plate"], data["parameters"]
    size = (max(164, fabric["width"] + 30), fabric["height"] + 48)
    pdf = new_pdf(path, size)
    label(pdf, 15, size[1] - 10, "Fabric web - cut 1", 5, bold=True)
    label(pdf, 15, size[1] - 16, "1:1 in mm | BLACK: cut | BLUE: mark | print at 100%", 2.7)
    sheet(pdf, fabric, 15, size[1] - 24)
    calibration(pdf, 15, 9)
    label(pdf, size[0] - 15, 16, "1 / 2", 2.5, align="end")
    size = (plate["width"] + 30, plate["height"] + 48)
    next_page(pdf, size)
    label(pdf, 15, size[1] - 10, f"Identical rigid plate - cut {plate['quantity']}", 5, bold=True)
    label(pdf, 15, size[1] - 16,
          f"1:1 in mm | plate radius {p['plate_radius']:g} mm | attachment R = {p['R']:g} mm", 2.7)
    sheet(pdf, plate, 15, size[1] - 24)
    calibration(pdf, 15, 9)
    label(pdf, size[0] - 15, 16, "2 / 2", 2.5, align="end")
    pdf.save()


def tile_grid(part):
    width, height, overlap, pad = 190, 250, 10, 10
    cols = max(1, ceil((part["width"] + 2 * pad - overlap) / (width - overlap)))
    rows = max(1, ceil((part["height"] + 2 * pad - overlap) / (height - overlap)))
    return width, height, overlap, pad, cols, rows


def cross(pdf, x, y):
    pdf.setStrokeColor(GRAY)
    pdf.setLineWidth(.12)
    pdf.line(x - 2, y, x + 2, y)
    pdf.line(x, y - 2, x, y + 2)
    pdf.circle(x, y, .65, stroke=1, fill=0)


def tiled(data, path):
    pdf = new_pdf(path, PAGE)
    label(pdf, 16, 280, "A4 cutting templates", 6, bold=True)
    paragraph(pdf, 16, 268,
              "Print at 100% / actual size. Disable Fit, Shrink and borderless scaling. "
              "Check the 100 mm bar before cutting. Black lines are cuts; blue lines "
              "are marks. Join each part's tiles separately using the matching crosses "
              "inside the 10 mm overlaps.", width=178)
    label(pdf, 16, 235, "Tile map", 4, bold=True)
    fgrid, pgrid = tile_grid(data["fabric"]), tile_grid(data["plate"])
    label(pdf, 16, 226, "Fabric: 1 piece", 3.5, bold=True)
    label(pdf, 101, 226, f"Plate: cut {data['plate']['quantity']} copies", 3.5, bold=True)
    for part, prefix, origin, diagram_width, diagram_height in (
            (data["fabric"], "F", (25, 217), 55, 147),
            (data["plate"], "P", (101, 217), 91, 95)):
        tw, th, over, pad, cols, rows = tile_grid(part)
        cw, ch = diagram_width / cols, diagram_height / rows
        for row in range(rows):
            for col in range(cols):
                px, py = origin[0] + col * cw, origin[1] - (row + 1) * ch
                pdf.setFillColor(PALE)
                pdf.setStrokeColor(GRAY)
                pdf.setLineWidth(.2)
                pdf.rect(px, py, cw - 1, ch - 1, stroke=1, fill=1)
                label(pdf, px + cw / 2, py + ch / 2 - 1,
                      f"{prefix}{row * cols + col + 1}", 3.2, align="middle")
    paragraph(pdf, 101, 108,
              f"Join all P tiles into one plate master. Use that master for all "
              f"{data['plate']['quantity']} identical plates.", width=90)
    paragraph(pdf, 16, 52,
              "Trim one sheet within each overlap, then align the paired crosses. "
              "Do not align paper edges. Keep the printed free-length gauge marks "
              "on the fabric. The assembly notes define their placement.", width=178)
    calibration(pdf, 16, 16)
    total = 1 + fgrid[4] * fgrid[5] + pgrid[4] * pgrid[5]
    page_number = 1
    label(pdf, 194, 12, f"1 / {total}", 2.5, align="end")
    for part, prefix, title in ((data["fabric"], "F", "Fabric web"),
                                (data["plate"], "P", "Rigid plate")):
        tw, th, over, pad, cols, rows = tile_grid(part)
        # Registration marks sit outside each part and in every shared strip.
        reg_x = [-5, part["width"] + 5]
        reg_y = [-5, part["height"] + 5]
        reg_x += [-pad + col * (tw - over) + over / 2 for col in range(1, cols)]
        reg_y += [-pad + row * (th - over) + over / 2 for row in range(1, rows)]
        for row in range(rows):
            for col in range(cols):
                next_page(pdf)
                page_number += 1
                tile = f"{prefix}{row * cols + col + 1}"
                label(pdf, 10, 285, f"{title} - {tile}", 4.5, bold=True)
                label(pdf, 10, 278, "PRINT 100% / ACTUAL SIZE - do not fit", 3, BLUE)
                label(pdf, 200, 278, f"row {row + 1}/{rows}, column {col + 1}/{cols}",
                      2.7, align="end")
                x0, y0 = -pad + col * (tw - over), -pad + row * (th - over)
                pdf.saveState()
                clip = pdf.beginPath()
                clip.rect(10, 20, tw, th)
                pdf.clipPath(clip, stroke=0, fill=0)
                sheet(pdf, part, 10 - x0, 270 + y0)
                for rx in reg_x:
                    for ry in reg_y:
                        if x0 <= rx <= x0 + tw and y0 <= ry <= y0 + th:
                            cross(pdf, 10 + rx - x0, 270 - ry + y0)
                pdf.restoreState()
                pdf.setStrokeColor(GRAY)
                pdf.setLineWidth(.1)
                pdf.rect(10, 20, tw, th)
                calibration(pdf, 10, 8)
                label(pdf, 200, 8, f"{tile} | {page_number} / {total}", 2.5, align="end")
    pdf.save()


def heading(pdf, number, title, subtitle):
    label(pdf, 16, 280, title, 6, bold=True)
    paragraph(pdf, 16, 269, subtitle, size=3.2)
    pdf.setStrokeColor(HexColor("#c7d0d6"))
    pdf.setLineWidth(.2)
    pdf.line(16, 19, 194, 19)
    label(pdf, 16, 12, "Six-path fabric actuator | dimensions in mm", 2.5, GRAY)
    label(pdf, 194, 12, f"{number} / 4", 2.5, GRAY, align="end")


def subsection(pdf, x, y, title, text, width=178):
    label(pdf, x, y, title, 3.8, bold=True)
    return paragraph(pdf, x, y - 7, text, width=width) - 5


def assembly(data, path):
    p, fabric, plate = data["parameters"], data["fabric"], data["plate"]
    outer, inner, length = p["R"], p["r"], p["L"]
    cells, plates = p["cells"], plate["quantity"]
    land, thickness = p["land"], p["plate_thickness"]
    grip_route = 2 * land + thickness
    h0 = sqrt(length * length - (outer - inner) ** 2)
    pdf = new_pdf(path, PAGE)
    heading(pdf, 1, "Cutting and part count",
            f"One connected fabric web and {plates} identical plates make the {cells}-cell stack. "
            "Adhesive fixes every fabric root; the slots alone do not anchor the legs.")
    label(pdf, 26, 243, "Cut 1 web", 3.6, bold=True)
    sheet(pdf, fabric, 33, 235, min(.105, 190/fabric["height"], 45/fabric["width"]), marks=False)
    label(pdf, 26, 43, f"{fabric['width']:.0f} x {fabric['height']:.2f}", 2.8)
    label(pdf, 89, 243, "Parts", 4, bold=True)
    y = paragraph(pdf, 89, 234,
                  f"1 fabric piece, nominal {p['fabric_thickness']:g} mm thick.\n"
                  f"{plates} rigid plates, nominal {thickness:g} mm thick.\n"
                  "Adhesive compatible with both materials.\n"
                  "No separate cords or individual fasteners.", width=105)
    y = subsection(pdf, 89, y - 7, "Geometry",
                   f"Outer attachment R = {outer:g}; inner r = {inner:g}; r/R = {inner/outer:.3g}. "
                   f"Plate radius = {p['plate_radius']:g}. Each free leg is {length:.3f} long. "
                   f"Untwisted free gap = {h0:.3f} per cell. Plate-center pitch = "
                   f"{p['initial_center_pitch']:.3f}; total outer height = "
                   f"{p['initial_total_height']:.2f}.", width=105)
    label(pdf, 89, y - 5, f"Plate master - cut {plates}", 3.6, bold=True)
    sheet(pdf, plate, 102, y - 13, min(.33, 92/plate["width"]))
    paragraph(pdf, 89, 52,
              "Black: through-cut. Blue: transfer marks. The five small U carriers "
              "connect the six rails only at the base. Keep them loose and outside "
              "the moving bays; they are not load paths.", width=105)

    next_page(pdf)
    heading(pdf, 2, "Identical plates, indexed roots",
            f"Number the cut plates 0 through {cells}. Viewed from the marked face, clock plate j "
            "clockwise by 60 degrees times j in the untwisted assembly. Keep root numbers visible.")
    sheet(pdf, plate, 17, 248, min(.48, 111/plate["width"]))
    label(pdf, 137, 241, "Root radii", 3.8, bold=True)
    paragraph(pdf, 137, 231,
              f"Roots 1, 3, 5: R = {outer:g}.\nRoots 2, 4, 6: r = {inner:g}.\n"
              "Successive roots are 60 degrees apart.\n"
              f"The plate outline is radius {p['plate_radius']:g}.\n"
              "Use the same plate file for every row.", width=58, size=3)
    paragraph(pdf, 137, 184,
              "Lane numbers stay on the fabric. At row j, route each lane through "
              "the root listed below. One row is one plate.", width=58, size=3)
    label(pdf, 16, 128, "Fabric lane -> local plate root", 4, bold=True)
    columns = [16, 43, 72, 94, 116, 138, 160, 182]
    for x, text in zip(columns, ("Plate", "Clock", "Lane 1", "Lane 2", "Lane 3",
                                 "Lane 4", "Lane 5", "Lane 6")):
        label(pdf, x, 119, text, 2.8, bold=True)
    row_height = min(6.05, 75/max(1, cells))
    row_font = min(2.8, row_height*.65)
    for row in data["rows"]:
        y = 110 - row["plate"] * row_height
        if row["plate"] % 2 == 0:
            pdf.setFillColor(PALE)
            pdf.rect(15, y - .25*row_height, 180, row_height, fill=1, stroke=0)
        values = [str(row["plate"]), f"{row['clock']:.0f} deg"]
        values += [str(root) for root in row["lane_to_root"]]
        for x, text in zip(columns, values):
            label(pdf, x, y, text, row_font)

    next_page(pdf)
    heading(pdf, 3, "Root routing and bonding",
            f"Each {p['grip']:g} mm marked grip goes out across one face, through its slot, "
            "and back across the opposite face. Set both free-length gauges at the root.")
    label(pdf, 16, 242, "Section through one plate and slot - schematic", 3.8, bold=True)
    pdf.setFillColor(HexColor("#dce3e8"))
    pdf.setStrokeColor(GRAY)
    pdf.setLineWidth(.2)
    pdf.rect(33, 167, 77, 15, fill=1, stroke=1)
    pdf.rect(117, 167, 59, 15, fill=1, stroke=1)
    pdf.setStrokeColor(BLUE)
    pdf.setLineWidth(1.1)
    for points in (((70, 225), (70, 183), (113.5, 183), (113.5, 166),
                    (70, 166), (70, 128)),):
        route = pdf.beginPath()
        route.moveTo(*points[0])
        for point in points[1:]:
            route.lineTo(*point)
        pdf.drawPath(route)
    pdf.setStrokeColor(HexColor("#cb7434"))
    pdf.setLineWidth(1.8)
    pdf.line(74, 182.2, 108, 182.2)
    pdf.line(74, 166.8, 108, 166.8)
    pdf.setStrokeColor(black)
    pdf.setLineWidth(.2)
    for y in (183, 166):
        pdf.line(65, y, 75, y)
    label(pdf, 70, 230, "free leg to next cell", 3, align="middle")
    label(pdf, 70, 120, "free leg to previous cell", 3, align="middle")
    label(pdf, 84, 190, f"{land:g} mm land", 3, align="middle")
    label(pdf, 84, 158, f"{land:g} mm land", 3, align="middle")
    label(pdf, 129, 173, f"{thickness:g} mm plate", 3)
    label(pdf, 121, 193, f"{p['slot_depth']:g} mm radial slot", 2.8)
    label(pdf, 25, 202, "root / gauge", 3)
    pdf.line(50, 199, 68, 185)
    label(pdf, 137, 151, "Orange: bond", 3, HexColor("#a65c26"))
    y = subsection(pdf, 16, 104, "The gauge fixes the geometry",
                   "Hold the two marked free-leg exit lines at the same root XY position "
                   "on opposite faces. Bond each land fully, including the exit edge, "
                   "so neither root peels away or slides. A slot by itself does not fix the root. "
                   "For back-face marks, use the mirrored plate_back_marked.svg after flipping "
                   "the plate left-to-right. Reference clocking always uses the front marked face.")
    paragraph(pdf, 16, y - 1,
              f"The nominal route uses {land:g} + {thickness:g} + {land:g} = {grip_route:g} mm "
              f"of the {p['grip']:g} mm grip. The remaining {p['fold_slack']:g} mm "
              "accommodates folds and finite fabric thickness. Check this on a coupon "
              "with your actual fabric, edge finish and adhesive before cutting the full web.")

    next_page(pdf)
    heading(pdf, 4, "Build and check",
            "The six free legs carry tension. The connecting carriers stay slack. "
            "This is a fabrication prototype, with no assigned load rating.")
    y = subsection(pdf, 16, 247, "1. Validate the root first",
                   "Make one slot-and-grip coupon. Transfer the gauge marks before threading. "
                   "Verify that the marked exits can coincide with the root on opposite faces. "
                   "Bond and test for slip, peel and tearing at the intended cord tension.")
    y = subsection(pdf, 16, y, "2. Cut and identify",
                   "Print actual size and measure the calibration bar. Cut the connected web "
                   "without severing its five base carriers. Orient the low-stretch yarn "
                   f"along the rails. Mark lanes 1-6 and rows 0-{cells}. "
                   f"Cut and deburr {plates} identical plates; smooth all fabric-contact edges.")
    y = subsection(pdf, 16, y, "3. Thread from the base",
                   "Use the row map on page 2. Feed each free rail end through its assigned "
                   "slot, then return it toward the marked root. Fit plates successively "
                   f"from row 0 to row {cells}. Keep the six rails separate and all "
                   f"{cells} free spans per rail untwisted.")
    y = subsection(pdf, 16, y, "4. Set the roots in an alignment jig",
                   f"Set the plate centers coaxially, with {h0:.3f} mm free gaps "
                   f"({p['initial_center_pitch']:.3f} mm center pitch) and each plate "
                   "clocked by its listed reference angle. Bond both face lands at every "
                   "marked gauge. The carrier loops belong loosely below the base. "
                   "Remove the temporary alignment jig only after the adhesive has cured.")
    y = subsection(pdf, 16, y, "5. Actuate and inspect",
                   "Apply an axial load that resists contraction and twist the ends. "
                   f"Each cell's angle is additional to its reference clocking. With {cells} "
                   f"uniform cells, 45 degrees per cell means {45*cells} degrees total end twist. "
                   "Check that all six legs stay taut and no fabric contacts, catches or slips.")
    y = subsection(pdf, 16, y, "Operating equations",
                   "h^2 = L^2 - R^2 - r^2 + 2 R r cos(theta).\n"
                   "For six equally loaded legs: T = F L / (6 h), and M = F R r sin(theta) / h.\n"
                   "Here F is the axial resisting load, T is one leg's tension, and M is torque.")
    paragraph(pdf, 16, y,
              "Stop at positive height: tension and required torque diverge as h approaches "
              "zero under load. Exact motion has been established for the six tensile "
              "centerlines. Finite-width fabric may twist or wrinkle; exact fabric isometry, "
              "adhesive durability and load capacity have not been validated.", color=BLUE)
    pdf.save()


def create_pdfs(layout_path, output_dir):
    data = json.loads(Path(layout_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "cutting_templates_full_size.pdf",
             output / "cutting_templates_A4.pdf", output / "strap_assembly.pdf"]
    for render, path in zip((full_size, tiled, assembly), paths):
        render(data, path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path,
                        default=Path(__file__).parent / "templates/default/layout.json")
    parser.add_argument("--output", type=Path, default=Path("output/pdf"))
    args = parser.parse_args()
    for path in create_pdfs(args.layout, args.output):
        print(path)


if __name__ == "__main__":
    main()
