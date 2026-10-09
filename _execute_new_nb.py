from __future__ import annotations

import os
import sys
from pathlib import Path

workspace = Path(r"C:\xampp1\Quantum")
notebook_path = workspace / sys.argv[1]
env_root = Path(r"C:\Users\tomqu\Documents\MLandPython\envs\helios_env")
os.environ["PATH"] = os.pathsep.join(
    (str(env_root), str(env_root / "Scripts"), os.environ.get("PATH", ""))
)
os.chdir(workspace)

import nbformat
from nbclient import NotebookClient

print(f"STARTING ML NOTEBOOK: {notebook_path.name}", flush=True)
print("Architecture unchanged; lambda in {0, 0.01}, T=2..4 step 0.2", flush=True)
notebook = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=7200,
    kernel_name="python3",
    resources={"metadata": {"path": str(workspace)}},
    allow_errors=False,
)
client.execute()
nbformat.write(notebook, notebook_path)
print(f"executed and saved {notebook_path}", flush=True)
