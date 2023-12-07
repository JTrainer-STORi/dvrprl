import subprocess
import pytest

def test_ppo():
    subprocess.run(
        "python dvrprl/ppo_cleanrl.py --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )