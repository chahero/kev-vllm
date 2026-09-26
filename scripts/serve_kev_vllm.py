"""Serve Kev with the current Python environment on Windows or Linux."""
import os
import subprocess
import sys
from kev_runtime import ROOT, server_command

if __name__ == "__main__":
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", OMP_NUM_THREADS="4")
    raise SystemExit(subprocess.call(server_command() + sys.argv[1:], cwd=ROOT, env=env))
