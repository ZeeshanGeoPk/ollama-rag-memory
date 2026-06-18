from __future__ import annotations

import shutil
import subprocess


def read_gpu_stats() -> dict:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {
            "available": False,
            "provider": None,
            "message": "NVIDIA GPU telemetry is unavailable.",
            "gpus": [],
        }

    query = (
        "name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    try:
        result = subprocess.run(
            [
                binary,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "provider": "nvidia",
            "message": f"Could not read GPU telemetry: {exc}",
            "gpus": [],
        }

    gpus = []
    for index, line in enumerate(result.stdout.splitlines()):
        values = [value.strip() for value in line.split(",")]
        if len(values) != 6:
            continue
        name, utilization, memory_used, memory_total, temperature, power = values
        gpus.append(
            {
                "index": index,
                "name": name,
                "utilization_percent": _number(utilization),
                "memory_used_mb": _number(memory_used),
                "memory_total_mb": _number(memory_total),
                "temperature_c": _number(temperature),
                "power_w": _number(power),
            }
        )
    return {
        "available": bool(gpus),
        "provider": "nvidia",
        "message": None if gpus else "No NVIDIA GPUs were reported.",
        "gpus": gpus,
    }


def _number(value: str) -> float | None:
    if value in {"N/A", "[Not Supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None
