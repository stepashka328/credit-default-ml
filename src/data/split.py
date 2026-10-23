import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42
df = pd.read_csv("data/raw/credit_default.csv")
target = "default.payment.next.month" if "default.payment.next.month" in df.columns else df.columns[-1]

initial, rest = train_test_split(df, train_size=0.6, stratify=df[target], random_state=SEED)
initial.to_csv("data/initial.csv", index=False)

Path("data/incoming").mkdir(exist_ok=True)
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
for i, (_, idx) in enumerate(skf.split(rest, rest[target]), 1):
    rest.iloc[idx].to_csv(f"data/incoming/batch_{i:02d}.csv", index=False)
    print(f"batch_{i:02d}: {len(idx)} строк")
print("initial:", len(initial), "строк")