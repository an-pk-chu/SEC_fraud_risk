"""Quick data-quality pass: class balance by year, feature missingness.
Run from the repo root: python src/01_explore.py
"""
import pandas as pd

df = pd.read_csv("data/data_FraudDetection_JAR2020.csv")
print("Shape:", df.shape)
print("\nYear range:", df.fyear.min(), "-", df.fyear.max())
print("\nOverall fraud rate: {:.3%}".format(df.misstate.mean()))
print("\nFirm-years by fyear (sample):")
print(df.groupby("fyear").misstate.agg(["count", "sum", "mean"]).tail(15))

feat_cols = ["dch_wc","ch_rsst","dch_rec","dch_inv","soft_assets","ch_cs",
             "ch_cm","ch_roa","issue","bm","dpi","reoa","EBIT","ch_fcf"]
print("\nMissingness in candidate features:")
print(df[feat_cols].isna().mean().sort_values(ascending=False))
