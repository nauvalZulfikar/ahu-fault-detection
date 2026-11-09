"""Reproduce the submission: python run.py  ->  ../predictions.csv"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ahu_fdd.pipeline import run  # noqa: E402

if __name__ == "__main__":
    sub, _, _ = run()
    print(sub["fault_type"].value_counts())
