import subprocess
import os
import sys

def main():
    script_path = os.path.join(os.path.dirname(__file__), "run_scgbinner.sh")

    subprocess.run([script_path] + sys.argv[1:])
