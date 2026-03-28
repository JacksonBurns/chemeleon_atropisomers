from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold

INPUT_PATH = "data.csv"
OUTPUT_DIR = Path("cv_splits")
OUTPUT_DIR.mkdir(exist_ok=True)
N_SPLITS = 5
RANDOM_STATE = 42
SHUFFLE = True

df = pd.read_csv(INPUT_PATH)

for repetition in range(5):
    kf = KFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE + repetition)
    subdir = OUTPUT_DIR / f"repetition_{repetition}"
    subdir.mkdir(exist_ok=True)
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        fold_dir = subdir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        val_frac = 0.125  # i.e. 70:10:20 split
        val_size = int(len(train_df) * val_frac)
        val_df = train_df.iloc[:val_size]
        train_df = train_df.iloc[val_size:]
        train_df.to_csv(fold_dir / "train.csv", index=False)
        val_df.to_csv(fold_dir / "val.csv", index=False)
        test_df.to_csv(fold_dir / "test.csv", index=False)
