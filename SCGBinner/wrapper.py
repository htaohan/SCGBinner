import subprocess
import os

def main():
    script_path = os.path.join(os.path.dirname(__file__), "run_scgbinner.sh")
    subprocess.run(["bash", script_path], check=True)