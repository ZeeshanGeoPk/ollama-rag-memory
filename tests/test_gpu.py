from __future__ import annotations

from ollama_middle_layer import gpu


def test_gpu_stats_reports_unavailable_without_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(gpu.shutil, "which", lambda _: None)

    result = gpu.read_gpu_stats()

    assert result["available"] is False
    assert result["gpus"] == []
