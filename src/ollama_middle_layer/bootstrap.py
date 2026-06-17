from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import socket
import subprocess
import time
from urllib.parse import urlparse


class OllamaBootstrapError(RuntimeError):
    pass


@dataclass
class ManagedServer:
    host_url: str
    process: subprocess.Popen | None = None
    owned: bool = False


def _host_port(host_url: str) -> tuple[str, int]:
    parsed = urlparse(host_url)
    if not parsed.hostname or not parsed.port:
        raise OllamaBootstrapError(f"Invalid Ollama host URL: {host_url}")
    return parsed.hostname, parsed.port


def is_tcp_reachable(host_url: str, timeout: float = 0.25) -> bool:
    host, port = _host_port(host_url)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class OllamaServerManager:
    def __init__(self, models_dir: Path, binary: str = "ollama") -> None:
        self.models_dir = models_dir
        self.binary = binary
        self.servers: list[ManagedServer] = []

    def start(self, host_url: str, timeout_seconds: float = 30.0) -> ManagedServer:
        if is_tcp_reachable(host_url):
            server = ManagedServer(host_url=host_url, owned=False)
            self.servers.append(server)
            return server

        binary_path = shutil.which(self.binary)
        if binary_path is None:
            raise OllamaBootstrapError(
                "Ollama binary was not found. Install Ollama first, then run this service again."
            )

        host, port = _host_port(host_url)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{host}:{port}"
        env["OLLAMA_MODELS"] = str(self.models_dir)

        process = subprocess.Popen(
            [binary_path, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        server = ManagedServer(host_url=host_url, process=process, owned=True)
        self.servers.append(server)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise OllamaBootstrapError(f"Ollama server exited early for {host_url}.")
            if is_tcp_reachable(host_url):
                return server
            time.sleep(0.25)

        process.terminate()
        raise OllamaBootstrapError(f"Timed out waiting for Ollama server on {host_url}.")

    def shutdown(self) -> None:
        for server in self.servers:
            if server.owned and server.process and server.process.poll() is None:
                server.process.terminate()
                try:
                    server.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.process.kill()


def _model_names(list_response: object) -> set[str]:
    names: set[str] = set()
    models = getattr(list_response, "models", None)
    if models is None and isinstance(list_response, dict):
        models = list_response.get("models", [])
    for model in models or []:
        name = getattr(model, "model", None) or getattr(model, "name", None)
        if name is None and isinstance(model, dict):
            name = model.get("model") or model.get("name")
        if name:
            names.add(str(name))
    return names


def ensure_model(client: object, model_name: str) -> bool:
    """Return True when a pull was needed."""
    installed = _model_names(client.list())
    if model_name in installed:
        return False
    client.pull(model_name)
    return True
