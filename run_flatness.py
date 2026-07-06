import subprocess, sys, os
app = r"C:\Users\user\Desktop\윤호\2026\RISE 학문후속세대 연구\floor_flatness_app.py"
os.chdir(os.path.dirname(app))
subprocess.run([sys.executable, "-m", "streamlit", "run", app])
