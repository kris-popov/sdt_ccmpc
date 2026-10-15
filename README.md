# Chance-Constrained NMPC using Covariance-based Signed-Distance Tightening

Course project for AEROSP 740 (Model Predictive Control, W26). 
The full writeup is in
[AE_740_MPC_Final_Project.pdf](AE_740_MPC_Final_Project.pdf).

## Repository layout

```
.
├── AE_740_MPC_Final_Project.pdf   # project report
├── requirements.txt
├── run_all.sh                     # runs all three evaluation cases
├── src/                           # Python implementation
│   ├── method.py                  # dynamics, LQR, covariance propagation, box SDF, NMPC solve
│   └── evaluate.py                # scenarios, closed-loop simulation, clearance check, plots
└── src_matlab/
    └── cc_nmpc.m                  # original MATLAB implementation (report appendix)
```

`src/method.py` contains the controller. `src/evaluate.py` runs the simulations.

## Running the Python version

This code requires Python 3.9 or newer. To recreate the results:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./run_all.sh
```

Or run each case individually:

```bash
python src/evaluate.py --case 1
python src/evaluate.py --case 2
python src/evaluate.py --case 3
```

Each run prints the distance to the goal at every step, the step at which the goal was reached, and the minimum clearance between the robot's
safety radius and any obstacle. It then outputs two figures:

- `results/case<N>_trajectory.png`: 3D trajectory with start (green), goal (magenta), and obstacles
- `results/case<N>_inputs.png`: thrust and body torques over time

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--case` | `1` | Scenario from the report (1, 2, or 3) |
| `--tsim` | `80` | Maximum number of closed-loop steps |
| `--horizon` | `20` | MPC horizon `N` |
| `--seed` | `2` | Seed for the process-noise generator |
| `--out` | `results` | Output directory for figures |
| `--quiet` | off | Suppress per-step output |

All other parameters (sampling time 0.1 s, cost weights, actuator and attitude limits, noise
covariance, risk level `alpha = 0.05`) correspond to the values from the report and are set in `run()` in
`src/evaluate.py`.

## Running the original MATLAB version

`src_matlab/cc_nmpc.m` is the script from the report appendix. It needs MATLAB with the Control
System Toolbox (for `dlqr`) and [CasADi for MATLAB](https://web.casadi.org/get/) on the path.
Case 1 is active by default; to run Case 2 or 3, comment out the Case 1 block and uncomment the
one you want (around line 50).
