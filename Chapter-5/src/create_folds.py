import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    #df = pd.read_csv("Chapter-5/input/cat_train.csv")
    df = pd.read_csv("Chapter-5/input/adult.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.income.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold_, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = fold_

    df.to_csv("Chapter-5/input/adult_folds.csv", index=False)
   # df.to_csv("Chapter-5/input/cat_train_folds.csv", index=False)