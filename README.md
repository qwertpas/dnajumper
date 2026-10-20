# DNA jumper

## GUI launch commands

Run these commands from the repository root. Click a screenshot to view it at full size.
Hardware screenshots show disconnected interfaces.

| GUI | Command | Screenshot |
| --- | --- | --- |
| Hunt truss buckling | `python3 -m trussbuckling` | [<img src="media/gui/hunt.png" alt="Hunt truss buckling" width="300">](media/gui/hunt.png) |
| Rigid folding prism and aspect ratio | `python3 -m trussbuckling.research.rigid_demo --sides 3` | [<img src="media/gui/rigid_prism.png" alt="Rigid folding prism and aspect ratio" width="300">](media/gui/rigid_prism.png) |
| Tensile-cord cells and stacks | `python3 -m trussbuckling.cable_gui` | [<img src="media/gui/cable.png" alt="Tensile-cord cells and stacks" width="300">](media/gui/cable.png) |
| Six-cell fabric assembly | `python3 -m trussbuckling.fabric_gui` | [<img src="media/gui/fabric.png" alt="Six-cell fabric assembly" width="300">](media/gui/fabric.png) |
| Three fabric strips without weave shear | `python3 -m trussbuckling.research.no_shear.gui` | [<img src="media/gui/no_shear.png" alt="Three fabric strips without weave shear" width="300">](media/gui/no_shear.png) |
| Spring-chain interactive 3D simulation | `python3 new_modeling/springchain.py --show` | [<img src="media/gui/springchain.png" alt="Spring-chain interactive 3D simulation" width="300">](media/gui/springchain.png) |
| DNA Jumper motor control | `python3 motor_gui_v2.py` | [<img src="media/gui/motor_gui_v2.png" alt="DNA Jumper motor control" width="300">](media/gui/motor_gui_v2.png) |
| Motor control + minDAQ | `python3 motor_mindaq_gui.py` | [<img src="media/gui/motor_mindaq_gui.png" alt="Motor control + minDAQ" width="300">](media/gui/motor_mindaq_gui.png) |
| Live force/torque | `python3 ft_live_gui.py` | [<img src="media/gui/ft_live_gui.png" alt="Live force/torque" width="300">](media/gui/ft_live_gui.png) |
| BLE stream test plots | `python3 ble_stream_plot.py` | [<img src="media/gui/ble_stream_plot.png" alt="BLE stream test plots" width="300">](media/gui/ble_stream_plot.png) |

The rigid prism also accepts `--radius 100 --sides 6 --aspect 1`.
The force/torque GUI accepts `--port /dev/cu.…`; run one application per serial port.
The combined motor/minDAQ GUI requires the external `mindaq_fw` checkout at the path configured in `motor_mindaq_gui.py`.
All GUIs use PyQt6. Install their dependencies with `python3 -m pip install -r requirements.txt`.
Scripts can also be launched by absolute path, for example `python3 /Users/chris/Code/dnajumper/motor_gui_v2.py`.

[Radius selection studies](studies/radius_selection/README.md) contains the current experiment sources, selected figures, numerical evidence, and reproduction instructions from the length sweep and radius separation chats.

Other modeling and measured-data analysis remain in [new_modeling](new_modeling/); firmware is in [gordonjumper](gordonjumper/).
