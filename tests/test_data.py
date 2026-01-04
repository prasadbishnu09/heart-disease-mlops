import pandas as pd
import os

def test_cleaned_data_exists():
    assert os.path.exists("data/processed/heart_clean.csv")

def test_data_shape():
    df = pd.read_csv("data/processed/heart_clean.csv")
    # 14 columns including target
    assert df.shape[1] == 14

def test_no_missing_values():
    df = pd.read_csv("data/processed/heart_clean.csv")
    assert df.isnull().sum().sum() == 0