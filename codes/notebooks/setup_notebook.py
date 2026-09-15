import os, sys, subprocess
from pathlib import Path


course_dir = Path("/content/drive/MyDrive/immerge_workshop")
os.chdir(course_dir)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-r", "codes/requirements.txt"], check=True)

print(course_dir)