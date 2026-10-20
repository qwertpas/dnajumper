"""Check the cut profiles, registered link lengths and manufacturing exports."""

from math import cos, pi, sqrt
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import ezdxf
import numpy as np
from shapely.geometry import LineString, Polygon

from trussbuckling.make_templates import Dimensions, back_marks, build, save_dxf, save_svg


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build(Dimensions())

    def test_one_blank_has_six_separate_active_rails_and_72_gauged_spans(self):
        data = self.data
        params = data["parameters"]
        shape = Polygon(data["fabric"]["cut"][0])
        self.assertTrue(shape.is_valid)
        self.assertEqual(len(shape.interiors), 0)
        np.testing.assert_allclose(shape.bounds, [0, 0, data["fabric"]["width"],
                                                  data["fabric"]["height"]])
        self.assertEqual(data["fabric"]["quantity"], 1)
        self.assertEqual(len(data["rows"]), 13)
        self.assertEqual(len(data["fabric"]["lines"]), 2*78)
        self.assertEqual(sum(t["text"].startswith("P") for t in data["fabric"]["text"]), 78)
        count = 0
        for lower, upper in zip(data["rows"][:-1], data["rows"][1:]):
            self.assertAlmostEqual(upper["y0"]-lower["y1"], params["L"])
            y = (lower["y1"]+upper["y0"])/2
            section = shape.intersection(LineString([(-1, y), (data["fabric"]["width"]+1, y)]))
            self.assertEqual(section.geom_type, "MultiLineString")
            self.assertEqual(len(section.geoms), 6)
            for rail in section.geoms:
                self.assertAlmostEqual(rail.length, params["rail"])
                count += 1
        self.assertEqual(count, 72)

    def test_registered_plate_roots_produce_equal_length_links(self):
        data = self.data
        params = data["parameters"]
        center = np.array([params["plate_radius"]]*2)
        local = np.array([[r["x"], r["y"]] for r in data["root_sites"]])-center
        for twist in (0., pi/6, 5*pi/12):
            h = sqrt(2*params["R"]*params["r"]*cos(twist))
            plates = []
            for j, row in enumerate(data["rows"]):
                angle = row["clock"]*pi/180+j*twist
                q = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
                roots = local[np.array(row["lane_to_root"])-1]@q.T
                expected_radii = np.where((np.arange(6)+j) % 2 == 0, params["R"], params["r"])
                expected_angles = np.arange(6)*pi/3+j*twist
                expected = expected_radii[:, None]*np.column_stack((np.cos(expected_angles),
                                                                    np.sin(expected_angles)))
                np.testing.assert_allclose(roots, expected, atol=1e-10)
                plates.append(np.column_stack((roots, np.full(6, j*h))))
            lengths = np.linalg.norm(np.diff(plates, axis=0), axis=2)
            np.testing.assert_allclose(lengths, params["L"], atol=1e-10)

    def test_plate_slots_are_inside_the_outline_with_remaining_rim(self):
        cuts = self.data["plate"]["cut"]
        outline = Polygon(cuts[0])
        self.assertTrue(Polygon(cuts[0], cuts[1:]).is_valid)
        self.assertEqual(len(cuts), 7)
        self.assertEqual(self.data["plate"]["quantity"], 13)
        self.assertEqual(outline.bounds, (0, 0, 230, 230))
        for cut in cuts[1:]:
            hole = Polygon(cut)
            self.assertTrue(outline.contains(hole))
            self.assertGreater(hole.distance(outline.boundary), 2)

    def test_base_carriers_have_slack_after_allowing_for_the_grip(self):
        data = self.data
        p = data["parameters"]
        # Everything at/above the first grip's bottom edge may be bonded.
        # Count only the remaining free U path, excluding its two inside bends.
        free = 2*data["rows"][0]["y0"]+p["carrier_gap"]-2*p["carrier_width"]
        self.assertAlmostEqual(free, p["carrier_length"])
        self.assertGreaterEqual(free-p["carrier_endpoint_bound"], p["carrier_slack"])
        self.assertAlmostEqual(free-p["carrier_endpoint_bound"],
                               data["verification"]["carrier_slack_lower_bound_mm"])
        shape = Polygon(data["fabric"]["cut"][0])
        y = data["rows"][0]["y0"]/2
        section = shape.intersection(LineString([(-1, y), (data["fabric"]["width"]+1, y)]))
        self.assertEqual(len(section.geoms), 10)
        for leg in section.geoms:
            self.assertAlmostEqual(leg.length, p["carrier_width"]+.3)

    def test_svg_and_dxf_round_trip_at_millimeter_scale(self):
        ns = {"svg": "http://www.w3.org/2000/svg"}
        with tempfile.TemporaryDirectory() as folder:
            for name in ("fabric", "plate"):
                sheet = self.data[name]
                svg_path = Path(folder)/f"{name}.svg"
                dxf_path = Path(folder)/f"{name}.dxf"
                save_svg(sheet, svg_path, marked=True)
                save_dxf(sheet, dxf_path)
                svg = ET.parse(svg_path).getroot()
                self.assertAlmostEqual(float(svg.attrib["width"][:-2]), sheet["width"], places=5)
                self.assertTrue(svg.attrib["width"].endswith("mm"))
                self.assertAlmostEqual(float(svg.attrib["height"][:-2]), sheet["height"], places=5)
                polygons = svg.findall("svg:g[@id='CUT']/svg:polygon", ns)
                self.assertEqual(len(polygons), len(sheet["cut"]))
                for polygon, expected in zip(polygons, sheet["cut"]):
                    actual = [[float(v) for v in point.split(",")]
                              for point in polygon.attrib["points"].split()]
                    np.testing.assert_allclose(actual, expected, atol=5.1e-7)
                doc = ezdxf.readfile(dxf_path)
                self.assertEqual(doc.units, 4)
                cuts = list(doc.modelspace().query('*[layer=="CUT"]'))
                if name == "plate":
                    circles = [entity for entity in cuts if entity.dxftype() == "CIRCLE"]
                    self.assertEqual(len(circles), 1)
                    self.assertEqual(circles[0].dxf.radius, 115)
                    expected = sheet["cut"][1:]
                else:
                    expected = sheet["cut"]
                paths = [entity for entity in cuts if entity.dxftype() == "LWPOLYLINE"]
                self.assertEqual(len(paths), len(expected))
                for entity, polygon in zip(paths, expected):
                    self.assertTrue(entity.closed)
                    actual = np.array(entity.get_points("xy"))
                    actual[:, 1] *= -1
                    np.testing.assert_allclose(actual, polygon[:-1], atol=1e-10)
                    self.assertTrue(Polygon(actual).is_valid)
                marks = list(doc.modelspace().query('*[layer=="MARK"]'))
                self.assertEqual(len(marks), len(sheet["lines"]))

    def test_back_face_marks_register_after_flipping_the_plate(self):
        front = self.data["plate"]
        back = back_marks(front)
        width = front["width"]
        for before, after in zip(front["lines"], back["lines"]):
            points = np.asarray(after["points"])
            points[:, 0] = width-points[:, 0]
            np.testing.assert_allclose(points, before["points"], atol=1e-12)
        for before, after in zip(front["text"][:6], back["text"][:6]):
            self.assertEqual(before["text"], after["text"])
            self.assertAlmostEqual(before["x"], width-after["x"])


if __name__ == "__main__":
    unittest.main()
