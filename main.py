import os
import sys
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

from utils import seed_everything
seed_everything(42)

def _set_env():
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    components_abs = os.path.join(get_original_cwd(), "components")
    os.environ["PYTHONPATH"] = (
        (os.environ.get("PYTHONPATH", "") + (os.pathsep if os.environ.get("PYTHONPATH") else ""))
        + components_abs
    )

def _abs_path(rel_path: str) -> str:
    """Path relative to the project root (not Hydra's run dir)."""
    return os.path.join(get_original_cwd(), rel_path)

def _parse_steps(cfg: DictConfig):
    steps = cfg.get("main", {}).get("steps", "all")
    if steps is None:
        steps = "all"
    if isinstance(steps, str):
        steps = steps.strip()
    if steps in ("all", ""):
        return ["download", "basic_cleaning", "data_split"]
    return [s.strip() for s in steps.split(",") if s.strip()]

def _get(cfg: DictConfig, path: str, default=None):
    try:
        node = cfg
        for p in path.split("."):
            node = node[p]
        return node
    except Exception:
        return default

@hydra.main(version_base=None, config_path=".", config_name="config")
def go(config: DictConfig):
    """
    Layout:
      - config.yaml in project root
      - components/get_data
      - src/basic_cleaning
      - src/data_split
    """
    _set_env()
    print("Resolved config:\n", OmegaConf.to_yaml(config))

    active_steps = _parse_steps(config)
    print("Active steps:", active_steps)

    # Absolute paths to each MLflow project
    comp_get_data   = _abs_path("components/get_data")
    comp_cleaning   = _abs_path("src/basic_cleaning")
    comp_data_split = _abs_path("src/data_split")

    # -----------------------
    # Step 1 — Download data
    # -----------------------
    if "download" in active_steps:
        sample = _get(config, "etl.sample", "sample1.csv")
        print(f"[download] sample={sample}")
        try:
            _ = mlflow.run(
                comp_get_data,
                entry_point="main",
                env_manager="local",
                parameters={
                    "sample": sample,
                    "artifact_name": "sample1.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )
        except Exception as e:
            print("[download] FAILED:", e, file=sys.stderr)
            raise

    # ----------------------------
    # Step 2 — Basic data cleaning
    # ----------------------------
    if "basic_cleaning" in active_steps:
        min_price = _get(config, "etl.min_price", 10)
        max_price = _get(config, "etl.max_price", 350)
        print(f"[basic_cleaning] min_price={min_price}, max_price={max_price}")
        try:
            _ = mlflow.run(
                comp_cleaning,
                entry_point="main",
                env_manager="local",
                parameters={
                    "input_artifact": "sample1.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "Data after basic cleaning",
                    "min_price": min_price,
                    "max_price": max_price,
                },
            )
        except Exception as e:
            print("[basic_cleaning] FAILED:", e, file=sys.stderr)
            raise

    # -------------------------
    # Step 3 — Split the data
    # -------------------------
    if "data_split" in active_steps:
        test_size   = _get(config, "modeling.test_size", 0.2)
        val_size    = _get(config, "modeling.val_size", 0.2)
        random_seed = _get(config, "modeling.random_seed", 42)
        stratify_by = _get(config, "modeling.stratify_by", "neighbourhood_group")
        print(f"[data_split] test_size={test_size}, val_size={val_size}, stratify_by={stratify_by}")
        try:
            _ = mlflow.run(
                comp_data_split,
                entry_point="main",
                env_manager="local",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "test_size": test_size,
                    "val_size": val_size,
                    "stratify_by": stratify_by,
                    "random_seed": random_seed,
                },
            )
        except Exception as e:
            print("[data_split] FAILED:", e, file=sys.stderr)
            raise

    print("Pipeline finished successfully ✅")

if __name__ == "__main__":
    go()
