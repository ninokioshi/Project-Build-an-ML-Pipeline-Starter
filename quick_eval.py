import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 1) Load the cleaned data (file is in your project root)
df = pd.read_csv("clean_sample.csv")

# 2) Define target and features
target = "price"
if target not in df.columns:
    raise SystemExit("Couldn't find 'price' column in clean_sample.csv")

# Common Airbnb features used in this project
numeric_cols = [
    "latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count", "availability_365"
]
categorical_cols = ["neighbourhood_group", "room_type"]
text_col = "name"  # short title of the listing

# Keep columns that actually exist in your file
numeric_cols = [c for c in numeric_cols if c in df.columns]
categorical_cols = [c for c in categorical_cols if c in df.columns]
use_text = text_col in df.columns

X = df[numeric_cols + categorical_cols + ([text_col] if use_text else [])].copy()
y = df[target].copy()

# 3) Split train/valid (match your config: test_size=0.2, seed=42, stratify by neighbourhood_group if present)
stratify = df["neighbourhood_group"] if "neighbourhood_group" in df.columns else None
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# 4) Build preprocessing
numeric_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if numeric_cols:
    transformers.append(("num", numeric_tf, numeric_cols))
if categorical_cols:
    transformers.append(("cat", categorical_tf, categorical_cols))
pre = ColumnTransformer(transformers=transformers, remainder="drop")

# Optional TF-IDF on the 'name' text (limit to 5 features like your config)
if use_text:
    text_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="")),
        ("tfidf", TfidfVectorizer(max_features=5))
    ])

    # ColumnTransformer can't apply TfidfVectorizer directly to a single Series,
    # so we do a small wrapper to concatenate it after the structured preprocessor.
    from sklearn.pipeline import FeatureUnion
    from sklearn.base import BaseEstimator, TransformerMixin

    class TextSelector(BaseEstimator, TransformerMixin):
        def __init__(self, key): self.key = key
        def fit(self, X, y=None): return self
        def transform(self, X): return X[self.key].astype(str).values

    text_pipe = Pipeline([("pick", TextSelector(text_col)), ("tfidf", TfidfVectorizer(max_features=5))])

    # FeatureUnion joins sparse/dense features; it works fine with RF
    features = FeatureUnion([("struct", pre), ("text", text_pipe)])
else:
    features = pre

# 5) RandomForest (match your config.yaml)
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=3,
    n_jobs=-1,
    criterion="squared_error",
    max_features=0.5,
    random_state=42
)

model = Pipeline([("features", features), ("rf", rf)])

# 6) Train & evaluate
model.fit(X_train, y_train)
preds = model.predict(X_valid)

mae = mean_absolute_error(y_valid, preds)
rmse = mean_squared_error(y_valid, preds, squared=False)
r2 = r2_score(y_valid, preds)

print("\n=== Evaluation on validation split ===")
print(f"MAE:  {mae:0.3f}")
print(f"RMSE: {rmse:0.3f}")
print(f"R^2:  {r2:0.3f}")
