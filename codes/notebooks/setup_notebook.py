import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    course_dir = Path("/content/drive/MyDrive/immerge_workshop")
    os.chdir(course_dir)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "codes/requirements.txt"], check=True)
else:
    course_dir = Path("/nfs/team292/projects/immerge_workshop")
    os.chdir(course_dir)

print(course_dir)