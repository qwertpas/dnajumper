# DC Motor System Identification Results

## Best Model: Second-Order with Voltage-Scaled Parameters

**R² = 0.9929** (fitted to rising portion of free-spin trials, V = 1-5V)

### Model Fit Comparison

![Final model comparison](sysid2_final.png)

*Blue = actual data, Magenta = scaled+delay model, Cyan = per-voltage fit, Red = basic 2nd order*

### Model Equations

```
State equations:
    wdot_state' = k * (a_eff * V - c_eff * w - wdot_state)
    w' = wdot_state

Parameter scaling:
    a_eff = a0 + a1/V = 12456 + 27941/V
    c_eff = c0 + c1/V = 54.2 + 144/V
    k = 296 1/s
```

This is equivalent to a second-order transfer function with voltage-dependent coefficients.

### Fitted Parameters

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| a0 | 12456 | rad/s²/V | Base torque/inertia ratio |
| a1 | 27941 | rad/s² | Torque scaling with 1/V |
| c0 | 54.2 | 1/s | Base mechanical damping |
| c1 | 144 | - | Damping scaling with 1/V |
| k | 296 | 1/s | Electrical dynamics (R/L) |

### Effective Parameters at Different Voltages

| V (volts) | a_eff | c_eff | τ_m (ms) | τ_e (ms) |
|-----------|-------|-------|----------|----------|
| 1.0 | 40397 | 198 | 5.0 | 3.4 |
| 2.0 | 26427 | 126 | 7.9 | 3.4 |
| 3.0 | 21768 | 102 | 9.8 | 3.4 |
| 4.0 | 19441 | 90 | 11.1 | 3.4 |
| 5.0 | 18044 | 83 | 12.0 | 3.4 |

Where:
- τ_e = 1/k = electrical time constant (inductance effect)
- τ_m = 1/c_eff = mechanical time constant

## Physical Interpretation

### Model Block Diagram

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
    V ──►[a_eff*V]──┼──►(+)──►[ 1/s ]──►[k]──►(+)──►[ 1/s ]──┼──► w
                    │    ▲              wdot      ▲           │
                    │    │                        │           │
                    │    └────[−c_eff]◄───────────┴───────────┘
                    │              ▲
                    │              │
                    │         [−1] (wdot feedback)
                    └─────────────────────────────────────────┘

    where: a_eff = 12456 + 27941/V
           c_eff = 54.2 + 144/V
           k = 296
```

### Why Second-Order?

The motor has **inductance** which causes current (and thus torque) to build up gradually rather than instantly. This creates the characteristic S-curve in the velocity response:

```
Electrical: V = L*di/dt + R*i + Ke*w
Mechanical: I*dw/dt = Kt*i - d*w
```

The electrical time constant τ_e = L/R ≈ 3.4ms represents how quickly current responds to voltage changes.

### Why Parameters Scale with 1/V?

At low voltages:
1. **Back-EMF is relatively more significant** - the ratio Ke*w/V is larger
2. **PWM effects** - at low duty cycles, there may be nonlinearities
3. **The system "feels" faster** - smaller steady-state velocity means the transient dominates

The 1/V scaling captures this behavior with a simple parameterization.

### Time Constants

- **Electrical (τ_e ≈ 3.4ms)**: How fast current builds up. Constant across voltages.
- **Mechanical (τ_m = 5-12ms)**: How fast velocity responds. Varies with voltage.

## How to Use This Model

### Python Implementation

```python
import numpy as np

# Model parameters
a0, a1 = 12456, 27941
c0, c1 = 54.2, 144
k = 296

def simulate_motor(V, t, w0=0):
    """
    Simulate motor velocity response to step voltage.

    Args:
        V: Applied voltage (volts)
        t: Time array (seconds)
        w0: Initial velocity (rad/s)

    Returns:
        w: Velocity array (rad/s)
    """
    a_eff = a0 + a1 / V
    c_eff = c0 + c1 / V

    w = np.zeros_like(t)
    w[0] = w0
    wdot_state = 0.0

    for j in range(1, len(t)):
        dt = t[j] - t[j-1]
        wddot = k * (a_eff * V - c_eff * w[j-1] - wdot_state)
        wdot_state = wdot_state + wddot * dt
        w[j] = w[j-1] + wdot_state * dt

    return w

# Example usage
t = np.linspace(0, 0.05, 500)  # 50ms
V = 3.0  # volts

w = simulate_motor(V, t)
w_steady = (a0 + a1/V) * V / (c0 + c1/V)  # Steady-state velocity
print(f"Steady-state velocity at {V}V: {w_steady:.1f} rad/s")
```

### Steady-State Predictions

At steady state (wdot = 0, wddot = 0):
```
w_ss = a_eff * V / c_eff = (a0 + a1/V) * V / (c0 + c1/V)
     = (a0*V + a1) / (c0 + c1/V)
```

| V | w_ss (rad/s) | w_ss (RPM) |
|---|--------------|------------|
| 1V | 204 | 1948 |
| 2V | 419 | 4001 |
| 3V | 639 | 6103 |
| 4V | 862 | 8231 |
| 5V | 1086 | 10371 |

### Estimating Rotor Inertia I

From the motor torque model, we know:
```
tau_stall / V = Kt/R ≈ 0.076 N·m/V  (fitted from sysid.py)
```

In our model, the torque/inertia relationship gives:
```
a_eff ≈ (Kt/R) / I
```

Using the per-voltage fits from sysid2.py:

| V | a_eff (fitted) | I = (Kt/R) / a_eff |
|---|----------------|---------------------|
| 3V | 21733 | 3.5e-6 kg·m² |
| 4V | 19380 | 3.9e-6 kg·m² |
| 5V | 17977 | 4.2e-6 kg·m² |

**Best estimate: I ≈ 3.9e-6 kg·m²** (average of mid-high voltage fits)

The variation with V suggests the 1/V scaling captures effects beyond pure inertia (possibly related to PWM behavior at low duty cycles).

### Converting to Physical Motor Parameters

With I ≈ 3.9e-6 kg·m²:
```
Kt/R = 0.076 N·m/V          (stall torque per volt)
Ke = 1/223.7 = 0.00447 V·s/rad  (back-EMF constant)
τ_e = L/R = 1/k = 3.4 ms    (electrical time constant)
```

At V=3V (mid-range):
```
a_eff = 21768 rad/s²/V  →  Kt/R = I * a_eff = 0.085 N·m/V
c_eff = 102 1/s         →  d_total/I = c_eff
                        →  d_total = 4.0e-4 N·m·s/rad
```

The total damping d_total includes both mechanical friction and electrical damping (Kt·Ke/R).

## Plots

### Raw vs Filtered Velocity Data

![Filtered data](sysid2_data.png)

*Blue dots = raw encoder velocity, Red line = savgol filtered (window=7, order=2)*

### Model Comparison

![Model comparison](sysid2_models.png)

*Comparison of different model structures: linear, quadratic, 2nd-order, delayed*

### Detailed Analysis

![Detailed comparison](sysid2_detailed.png)

*Per-voltage fits (cyan) show that parameters vary with voltage*

## Data and Code

- Raw data: `motor_logs/*free*.csv`
- Analysis script: `sysid2.py`

## Notes

1. Model was fit to **rising portion only** (0-95% of max velocity)
2. Low voltage trials (V < 1V) were excluded due to noise
3. The savgol filter (window=7, order=2) was used to smooth velocity data
4. The 1/V scaling may break down at very low or very high voltages
