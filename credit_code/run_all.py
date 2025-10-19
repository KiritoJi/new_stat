# run_all.py

import subprocess, sys

for name in ["train_logistic.py", "train_extratrees.py", "train_xgboost.py", "train_linear_regression.py"]:
    print(f"\n=== Running {name} ===")
    subprocess.run([sys.executable, name], check=False)