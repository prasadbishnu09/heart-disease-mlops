import pandas as pd

def load_and_clean_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]

    df = pd.read_csv(url, names=cols)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv("data/processed/heart_clean.csv", index=False)
    return df

if __name__ == "__main__":
    load_and_clean_data()