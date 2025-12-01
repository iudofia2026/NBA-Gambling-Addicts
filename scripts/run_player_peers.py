#!/usr/bin/env python3
"""Bootstrap a local venv and run the Player Peers Streamlit app.

This script will create `.venv` in the repo root (if missing), install `requirements.txt`,
and then exec `streamlit run apps/player_peers/app.py` using the venv Python. Use this when
your system Python environment has incompatible binary packages (e.g., NumPy 2 vs compiled
extensions built for NumPy <2).
"""
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"
REQ_FILE = REPO_ROOT / "requirements.txt"


def create_venv(venv_path: Path):
    print(f"Creating virtualenv at {venv_path}...")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])


def install_requirements(python_exe: str, req_file: Path):
    print(f"Installing requirements from {req_file} into {python_exe}...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([python_exe, "-m", "pip", "install", "-r", str(req_file)])


def run_streamlit(python_exe: str):
    cmd = [python_exe, "-m", "streamlit", "run", "apps/player_peers/app.py"]
    print("Launching Streamlit:", " ".join(cmd))
    os.execv(cmd[0], cmd)


def main():
    os.chdir(str(REPO_ROOT))
    if not REQ_FILE.exists():
        print("requirements.txt not found in repo root. Please create it or install dependencies manually.")
        sys.exit(1)

    if not VENV_DIR.exists():
        create_venv(VENV_DIR)

    # determine venv python
    if sys.platform == "win32":
        python_exe = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_exe = VENV_DIR / "bin" / "python"

    python_exe = str(python_exe)

    install_requirements(python_exe, REQ_FILE)

    # exec streamlit under venv python
    run_streamlit(python_exe)


if __name__ == "__main__":
    main()
