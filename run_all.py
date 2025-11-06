# run_all.py — one-click runner to fix env, set offline, place sample.csv, and run the pipeline
import os, sys, shutil, subprocess

def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg])

# 1) Make sure key deps exist (idempotent)
for p in ["mlflow", "hydra_core", "wandb"]:
    ensure(p.replace("_", "-"))

# 2) Keep W&B offline so the code reads local files deterministically
os.environ["WANDB_MODE"] = "offline"

# 3) Find sample1.csv anywhere in the project
candidates = []
for root, dirs, files in os.walk(".", topdown=True):
    if "sample1.csv" in files:
        candidates.append(os.path.join(root, "sample1.csv"))

if not candidates:
    print("ERROR: Could not find sample1.csv in this project. Run `python main.py` once to let get_data create it, then re-run this script.")
    sys.exit(1)

src = sorted(candidates, key=len)[0]  # pick the first/shortest path
print(f"Using: {src}")

# 4) Put sample.csv where basic_cleaning expects it when offline
os.makedirs("src/basic_cleaning", exist_ok=True)
shutil.copyfile(src, "src/basic_cleaning/sample.csv")
# also put one at project root for good measure
shutil.copyfile(src, "sample.csv")

# 5) Run the full pipeline
print("Running main.py ...")
subprocess.check_call([sys.executable, "main.py"])
