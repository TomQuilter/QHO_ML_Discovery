"""Overnight scan. No console window, so it cannot pause for a keypress.

Run it with:

    C:\\Users\\tomqu\\Documents\\MLandPython\\envs\\helios_env\\python.exe -u overnight_lam0p3_0p5.py

That starts the scan in the background and returns at once. Progress is written to
overnight_lam0p3_0p5.log. Run the same command again to continue a stopped scan.
If a scan is already running, the new one refuses to start.

Order:
1. Complex_Time_MAIN_QHO_Anharmonic_MU_Multihimble_overnight.ipynb
2. Complex_Time_QHO_Anharmonic_MU_lambda0p1_T3p6_5thimble_overnight.ipynb

lambda = 0.3 and 0.5, T = 0 to 6, step 0.05.
A failed time is logged and the scan moves on.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "overnight_lam0p3_0p5.log"
PIDFILE = ROOT / "overnight_lam0p3_0p5.pid"
HELIOS = Path(r"C:\Users\tomqu\Documents\MLandPython\envs\helios_env\python.exe")
NOTEBOOKS = [
    ROOT / "Complex_Time_MAIN_QHO_Anharmonic_MU_Multihimble_overnight.ipynb",
    ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda0p1_T3p6_5thimble_overnight.ipynb",
]
# Detached, new process group, no console window.
DETACHED = 0x00000008 | 0x00000200 | 0x08000000


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def running_pid() -> int | None:
    if not PIDFILE.exists():
        return None
    try:
        pid = int(PIDFILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    if process_alive(pid):
        return pid
    return None


def exec_notebook(path: Path) -> None:
    os.chdir(ROOT)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import nbformat

    nb = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    print(f"\n======== {path.name} ========", flush=True)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        lines = [
            line for line in cell.source.splitlines()
            if not line.lstrip().startswith("%")
        ]
        source = "\n".join(lines)
        if not source.strip():
            continue
        print(f"\n----- {path.name} cell {index} -----", flush=True)
        exec(compile(source, f"{path.name}:cell{index}", "exec"), namespace)
    print(f"\n======== finished {path.name} ========", flush=True)


def run_scan() -> None:
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    print("Overnight scan: lambda 0.3 and 0.5, T = 0 to 6, step 0.05.", flush=True)
    print("No console. This process keeps going without a keypress.", flush=True)
    print(f"Log: {LOG}", flush=True)
    try:
        for path in NOTEBOOKS:
            if not path.exists():
                print(f"Missing notebook: {path}", flush=True)
                continue
            try:
                exec_notebook(path)
            except Exception:
                traceback.print_exc()
                print(f"{path.name} stopped. Continuing with the next notebook.", flush=True)
        print("Scan finished.", flush=True)
        print("MAIN table: runs/overnight_MAIN_lam0p3_0p5_T0to6_dt0p05_N16/multihimble_vs_lattice.csv", flush=True)
        print("5-thimble table: runs/overnight_5thimble_lam0p3_0p5_T0to6_dt0p05_N16/five_thimble_vs_lattice.csv", flush=True)
    finally:
        if PIDFILE.exists() and PIDFILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
            PIDFILE.unlink()


def launch() -> int:
    pid = running_pid()
    if pid is not None:
        print(f"Already running (pid {pid}). Log: {LOG}")
        return 0
    python = HELIOS if HELIOS.exists() else Path(sys.executable)
    log = LOG.open("a", encoding="utf-8", buffering=1)
    log.write("\n--- starting detached scan ---\n")
    log.flush()
    subprocess.Popen(
        [str(python), "-u", str(Path(__file__).resolve()), "--run"],
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=DETACHED,
        close_fds=True,
    )
    print(f"Started. It will keep running with no window. Log: {LOG}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--run":
        run_scan()
    else:
        raise SystemExit(launch())
