# setup_translate_env.py
import os
import subprocess
import sys

env_name = "translate-env"
python_path = sys.executable
env_dir = os.path.join(os.getcwd(), env_name)

# Step 1: Create venv if it doesn't exist
if not os.path.exists(env_dir):
    print(f"Creating virtual environment: {env_name}")
    subprocess.run([python_path, "-m", "venv", env_name])
else:
    print(f"Virtual environment {env_name} already exists.")

# Step 2: Install required packages
pip_path = os.path.join(env_dir, "Scripts", "pip.exe")
requirements = ["transformers", "torch"]

print("Installing required packages...")
subprocess.run([pip_path, "install"] + requirements)

print("\n✅ Environment setup done!")
print(f"👉 To activate it, run:\n   {env_name}\\Scripts\\activate")
print(f"👉 To run your script:\n   python app.py")
