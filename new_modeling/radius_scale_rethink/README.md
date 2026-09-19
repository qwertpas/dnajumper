# Radius scale/model check

Run:

```bash
python3.11 new_modeling/radius_scale_rethink/analyze.py
```

The script leaves all original tracking files unchanged. It:

- measures the 46 mm outer fixture height in the final video frames;
- calculates one camera scale for each radius dataset;
- writes rescaled tracking CSVs under `results/tracking_csv`;
- compares corrected maximum velocity with a reduced spring-chain model;
- uses the final smooth 4 V torque-speed law from
  `/Users/chris/Code/flywheeljumper/main/firmware/forcetorquemotor2/data/logs4_highspeed_defaultservo/motormodel.ipynb`;
- fits a diagnostic downstream dry-friction torque while keeping the motor model fixed.

The median fixture measurements are 494, 496, 532, and 514 pixels for r5,
r7, r9, and r11. These correspond to 10,739, 10,783, 11,565, and 11,174
px/m. Correcting the camera scale causes r5 to overtake r9 in the high-mass
trials, including 180 g and 190-220 g.
