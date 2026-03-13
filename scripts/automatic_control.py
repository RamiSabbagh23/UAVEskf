from __future__ import annotations

import sys
import time
from pathlib import Path

import airsim

ROOT = Path(__file__).resolve().parents[1]


from fusionUAV.runtime.automatic_control import AutomaticControlRunner
from fusionUAV.runtime.eskf_runtime import load_app_config


def main() -> None:
    config_path = ROOT / "configs" / "eskf_config.yaml"
    app_cfg = load_app_config(config_path)

    client = airsim.MultirotorClient()
    client.confirmConnection()

    controller = AutomaticControlRunner(
        client=client,
        vehicle_name=app_cfg.vehicle.name,
        cfg=app_cfg.automatic_control,
    )
    controller.start()

    try:
        while not controller.stop_requested:
            controller.update()
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
