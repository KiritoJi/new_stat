# run_all.py
import subprocess, sys

print("=== Logistic Regression ===")
subprocess.run([sys.executable, "train_logistic.py"], check=False)

print("\n=== Extra Trees ===")
subprocess.run([sys.executable, "train_extratrees.py"], check=False)

print("\n=== XGBoost ===")
subprocess.run([sys.executable, "train_xgboost.py"], check=False)
