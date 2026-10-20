"""Six-path, zero-carrier-shear audit for three proposed serpentine strips.

This checks the load-path mechanics only. Every intermediate clamp is assumed
to prevent material sliding, so its two adjacent spans may have different
tensions. Rotating attachment hardware and unloaded connections are not
modeled. EA is 500 N per span, or 1000 N for a paired up/down set; one physical
serpentine ribbon with uniform section therefore has EA=500 N.
"""

import json
from pathlib import Path

from scipy.optimize import brentq

from ..manufacturable_panels.alignment_check import aligned, imperfect


def main():
    nominal, errors = [], []
    for angle in (45, 60, 75):
        nominal.append(aligned(angle, 0)[0])
        for shift, tilt in ((5, 0), (0, 1), (5, 1)):
            row = imperfect(angle, shift, tilt, shear=0)
            errors.append(row)
            print(row, flush=True)
    critical = brentq(lambda angle: aligned(angle, 0)[0]['minimum_full'], 60, 89, xtol=1e-7)
    report = dict(cells=6, force_N=40, radius_mm=100, inner_radius_mm=65,
                  EA_per_span_N=500, EA_per_up_down_pair_N=1000, anchor_family_offset_degrees=28.9,
                  boundary='Bottom fixed. Top x offset and rotation about y prescribed, total twist held, top z free; five internal plates fully free.',
                  assumptions='Zero cloth shear; straight elastic tensile spans with fixed natural lengths, all span clamps nonsliding. Fabric geometry, swivel mechanics, thickness, contact and strength are not represented.',
                  nominal=nominal, imperfections=errors, nominal_critical_angle=critical)
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Nominal critical angle:', critical)


if __name__ == '__main__':
    main()
