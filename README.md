# fusionUAV

`fusionUAV` is a Python workspace for AirSim-based UAV state estimation with an error-state Kalman filter (ESKF). It includes live sensor fusion, optional manual or automatic control, live plotting, and run logging/export.
This project was developed as part of a university project for the course "Selected Topics in Robotics and AI".

## Features

- Live ESKF estimation from IMU, GPS, barometer, and magnetometer data
- Keyboard-based manual control with optional command recording
- Automatic control from configured steps or replayed manual recordings
- Real-time visualization in `pyqtgraph` and optional trajectory plotting in AirSim
- CSV logging and PNG export for completed runs

## Repository Layout

```text
configs/
  eskf_config.yaml          Main runtime configuration
  settings.json             Example AirSim settings file

scripts/
  main.py                   Main live runtime
  automatic_control.py      Automatic control runner
  eskf.py                   Legacy standalone ESKF script
  manual_control.py         Legacy standalone manual control script

src/fusionUAV/
  eskf/                     ESKF core implementation
  runtime/                  Runtime, config loading, control, logging
  sensors/                  AirSim sensor readers and sensor config builders
  visualization/            Live plotting and AirSim plotting helpers
  utils/                    Shared math/utility helpers
```

Run artifacts are written under `outputs/` at runtime:

- `outputs/live_eskf/<timestamp>/` for state logs and plots
- `outputs/control/` for recorded manual command sequences

## Requirements

- Python 3.10 or newer
- AirSim running locally
- A multirotor vehicle and sensors configured in AirSim `settings.json`
- Windows is the expected environment for keyboard manual control

The default config expects the AirSim settings file at `~/Documents/AirSim/settings.json`. A sample file is included at [configs/settings.json](configs/settings.json).

## Installation

Install the package in editable mode before running the scripts:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install airsim --no-build-isolation
python -m pip install -e .
```

PowerShell helper:

```powershell
.\install.ps1
```

## Configuration

Main configuration file:

- [configs/eskf_config.yaml](configs/eskf_config.yaml)

Important sections:

- `paths.airsim_settings`: Path to the AirSim `settings.json` file
- `vehicle.name`: AirSim vehicle name, for example `Drone1`
- `sensors`: Enable or disable GPS, barometer, and magnetometer updates
- `periods`: Minimum update periods for IMU, GPS, barometer, and magnetometer
- `initial_covariance`: Initial state covariance standard deviations
- `control.mode`: `manual`, `automatic`, or `off`
- `manual_control`: Keyboard bindings, motion settings, and recording options
- `automatic_control`: Replay settings and fallback automatic steps
- `printing`: Console print interval for state output
- `plotting`: AirSim and live plot options

Relative paths such as `manual_control.record_path` and `automatic_control.recorded_commands_path` are resolved relative to the config file location.

## Control Modes

### Manual Control

Use:

```yaml
control:
  mode: "manual"
```

Manual control is implemented in [src/fusionUAV/runtime/manual_control.py](src/fusionUAV/runtime/manual_control.py).

Default actions:

- `T`: takeoff
- `L`: land
- `Esc`: emergency land and stop
- `W/S`: forward/backward
- `A/D`: left/right strafe
- `Up/Down`: rise/sink
- `Left/Right`: yaw left/right

If `manual_control.record_enable: true`, the runtime records the executed command sequence and saves it to `outputs/control/last_manual_control.yaml` by default when the run stops.

### Automatic Control

Use:

```yaml
control:
  mode: "automatic"
```

Automatic control is implemented in [src/fusionUAV/runtime/automatic_control.py](src/fusionUAV/runtime/automatic_control.py).

It can run in either of these modes:

1. Replay a recorded manual command file when `automatic_control.use_recorded_commands: true`.
2. Fall back to the configured `automatic_control.steps` sequence when no recording is available.

Supported automatic step modes:

- `velocity`
- `hover`
- `takeoff`
- `land`
- `attitude`

`mode` defaults to `velocity` if it is omitted in a step.

Example:

```yaml
automatic_control:
  steps:
    - name: "forward"
      mode: "velocity"
      vx: 2.0
      vy: 0.0
      vz: 0.0
      yaw_rate: 0.0
      duration: 3.0
```

## Running

### Main Runtime

Entry point: [scripts/main.py](scripts/main.py)

```powershell
python scripts/main.py
```

The main runtime:

- resets AirSim before startup
- waits for fresh post-reset GPS/ground-truth data
- runs the live ESKF loop
- optionally starts manual or automatic control
- updates live plots and optional AirSim trajectory plots
- logs state history and exports CSV/PNG artifacts on shutdown

### Automatic Control Only

Entry point: [scripts/automatic_control.py](scripts/automatic_control.py)

```powershell
python scripts/automatic_control.py
```

### Legacy Scripts

These older standalone scripts are still present for reference and quick experiments:

- [scripts/eskf.py](scripts/eskf.py)
- [scripts/manual_control.py](scripts/manual_control.py)

## Outputs

When the main runtime exits cleanly, it writes a timestamped folder under:

```text
outputs/live_eskf/YYYYMMDD_HHMMSS/
```

Artifacts are generated by [src/fusionUAV/runtime/run_logger.py](src/fusionUAV/runtime/run_logger.py).

Typical files include:

- `state_log.csv`
- `position_est_vs_gt.png`
- `velocity_est_vs_gt.png`
- `quaternion_est_vs_gt.png`
- `accel_bias_est.png`
- `gyro_bias_est.png`
- `gravity_vector_est.png`
- `live_position_plot.png`

The CSV contains time, estimated position/velocity/quaternion, ground-truth position/velocity/quaternion, estimated accelerometer bias, estimated gyroscope bias, and the estimated gravity vector. Ground-truth position is aligned to the filter's GPS/home NED frame before logging.

Manual recordings are stored under `outputs/control/`, with `last_manual_control.yaml` used by default for automatic replay.

## Sensor Configuration

Sensor noise settings are built from the AirSim `settings.json` file by [src/fusionUAV/sensors/build_from_settings.py](src/fusionUAV/sensors/build_from_settings.py).

The builder reads:

- IMU white noise and bias random walk settings
- GPS position covariance
- Barometer altitude noise
- Magnetometer covariance and reference field

It expects the AirSim sensor names `Imu`, `Gps`, `Barometer`, and `Magnetometer` under the configured vehicle.

## Development Notes

- Package metadata and runtime dependencies live in [pyproject.toml](pyproject.toml).
- Additional pinned dependencies are listed in [requirements.txt](requirements.txt).
- There is no committed `tests/` directory in the current repository snapshot.
