import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("ccpp.csv")

# Section 3: Data-quality checks
print("--- Section 3 ---")
print(f"Rows: {len(df)}, Columns: {df.shape[1]} {list(df.columns)}")
print(f"Missing values: {int(df.isna().sum().sum())}")
print(f"Exact duplicate rows: {int(df.duplicated().sum())}")
for col in df.columns:
    print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}")

# Section 4: Summary statistics and correlation
print("\n--- Section 4 ---")
for col in ["AT", "PE"]:
    print(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, std={df[col].std():.2f}")
corrs = df.corr()["PE"].drop("PE")
print("Correlations with PE:", ", ".join(f"{k}={v:.3f}" for k, v in corrs.items()))

# Section 5: Models and test error (first 7654 train, last 1914 test)
train = df.iloc[:7654]
test = df.iloc[7654:]
print("\n--- Section 5 ---")
print(f"Split: {len(train)} train, {len(test)} test")

base_mean = train["PE"].mean()
base_mae = mean_absolute_error(test["PE"], np.full(len(test), base_mean))
print(f"Baseline: train_mean={base_mean:.2f} MW, test_MAE={base_mae:.2f} MW")

lr_at = LinearRegression().fit(train[["AT"]], train["PE"])
at_mae = mean_absolute_error(test["PE"], lr_at.predict(test[["AT"]]))
print(f"AT model: test_MAE={at_mae:.2f} MW")

lr_v = LinearRegression().fit(train[["V"]], train["PE"])
a, b = lr_v.coef_[0], lr_v.intercept_
v_mae = mean_absolute_error(test["PE"], lr_v.predict(test[["V"]]))
print(f"V model: a={a:.3f} MW/cm Hg, b={b:.2f} MW, test_MAE={v_mae:.2f} MW")

row = df.iloc[7654]
pred = a * row["V"] + b
err = abs(row["PE"] - pred)
print(f"Row 7656: V={row['V']:.2f} cm Hg, pred={pred:.2f} MW, true={row['PE']:.2f} MW, abs_err={err:.2f} MW")

# Versions for 6.4
print("\n--- Versions ---")
print(f"Python {sys.version.split()[0]}, pandas {pd.__version__}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}")
