import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def run(folds: int):
    # Read data
    df = pd.read_csv("./input/train.csv")

    # Create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # Randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Fetch targets
    y = df.Survived.values

    # Initiate the kfold class from model_selection module
    kf = StratifiedKFold(n_splits=folds)

    # Fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # Save the new csv with kfold column
    df.to_csv("./input/train_folds.csv", index=False)

if __name__ == "__main__":
    run(folds=5)