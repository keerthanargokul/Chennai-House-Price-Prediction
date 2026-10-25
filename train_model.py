
from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. Load dataset
DATA_PATH = Path("data") / "Chennai houseing sale.csv"
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 2. Basic cleaning
df = df.drop_duplicates()
df.columns = df.columns.str.strip()

# Clean whitespace in categorical columns
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].str.strip()

# Correct common spelling variations
area_fixes = {
    "Adyr": "Adyar",
    "Ana Nagar": "Anna Nagar",
    "Chrompt": "Chrompet",
    "Velchery": "Velachery",
    "Kknagar": "KK Nagar",
    "Tnagar": "T Nagar"
}

df["AREA"] = df["AREA"].replace(area_fixes)

# Remove rows where the target is missing, if any
df = df.dropna(subset=["SALES_PRICE"])

# 3. Select features and target
# Exclude identifiers, transaction-related amounts and raw dates
drop_cols = [
    "SALES_PRICE", "PRT_ID", "REG_FEE", "COMMIS",
    "DATE_SALE", "DATE_BUILD"
]

X = df.drop(columns=drop_cols)
y = df["SALES_PRICE"]

# 4. Identify numeric and categorical features
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumeric features:", numeric_cols)
print("Categorical features:", categorical_cols)

# 5. Handle missing values and encode categories
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])


# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 7. Compare baseline models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    )
}

results = []
trained_models = {}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5
    })

    trained_models[name] = pipeline

# 8. Tune Random Forest using cross-validation
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42,
        n_jobs=1
    ))
])

param_grid = {
    "model__n_estimators": [50, 100,150],
    "model__max_depth": [10,20, None],
    "model__min_samples_leaf": [ 2, 5, 10],
    "model__max_features": [1,2]
}

search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=2,
    verbose=1,
    refit=True
)

print("\nTuning Random Forest with 5-fold cross-validation...")
search.fit(X_train, y_train)

tuned_rf = search.best_estimator_
rf_predictions = tuned_rf.predict(X_test)

results.append({
    "Model": "Tuned Random Forest",
    "R2": r2_score(y_test, rf_predictions),
    "MAE": mean_absolute_error(y_test, rf_predictions),
    "RMSE": mean_squared_error(y_test, rf_predictions) ** 0.5
})

trained_models["Tuned Random Forest"] = tuned_rf

print("\nBest Random Forest parameters:")
print(search.best_params_)
print(
    "Best cross-validation MAE: ₹",
    round(-search.best_score_, 2)
)

# 9. Display results and save a separate candidate
results_df = pd.DataFrame(results).sort_values(
    by="MAE", ascending=True
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

# IMPORTANT: do not overwrite the existing model.pkl
Path("model").mkdir(exist_ok=True)

candidate_path = Path("model") / "model_tuned_candidate.pkl"
joblib.dump(best_model, candidate_path)

print("\nBest model on held-out test set:", best_name)
print("Saved experimental model to:", candidate_path)
print("Existing model/model.pkl was not changed.")
print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")


# ==========================================
# 10. VISUALIZATION AND MODEL ANALYSIS
# ==========================================

from pathlib import Path
import numpy as np

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Get predictions from the best model
best_predictions = best_model.predict(X_test)

# ------------------------------------------
# GRAPH 1: MODEL COMPARISON
# ------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(
    data=results_df,
    x="R2",
    y="Model",
    ax=axes[0]
)
axes[0].set_title("Model Comparison - R² Score")
axes[0].set_xlabel("R² Score (higher is better)")
axes[0].set_ylabel("Model")
axes[0].set_xlim(0, 1)

sns.barplot(
    data=results_df,
    x="MAE",
    y="Model",
    ax=axes[1]
)
axes[1].set_title("Model Comparison - MAE")
axes[1].set_xlabel("Mean Absolute Error (₹)")
axes[1].set_ylabel("Model")

fig.tight_layout()
fig.savefig(output_dir / "model_comparison.png", dpi=300)
plt.close(fig)

# ------------------------------------------
# GRAPH 2: ACTUAL VS PREDICTED PRICES
# ------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    y_test,
    best_predictions,
    alpha=0.5
)

minimum = min(y_test.min(), best_predictions.min())
maximum = max(y_test.max(), best_predictions.max())

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect predictions"
)

ax.set_title(f"Actual vs Predicted Prices - {best_name}")
ax.set_xlabel("Actual Sale Price (₹)")
ax.set_ylabel("Predicted Sale Price (₹)")
ax.legend()

fig.tight_layout()
fig.savefig(output_dir / "actual_vs_predicted.png", dpi=300)
plt.close(fig)

# ------------------------------------------
# GRAPH 3: RESIDUAL PLOT
# Residual = Actual - Predicted
# ------------------------------------------

residuals = np.asarray(y_test) - best_predictions

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    best_predictions,
    residuals,
    alpha=0.5
)

ax.axhline(
    y=0,
    linestyle="--"
)

ax.set_title(f"Residual Plot - {best_name}")
ax.set_xlabel("Predicted Sale Price (₹)")
ax.set_ylabel("Residual (Actual - Predicted) (₹)")

fig.tight_layout()
fig.savefig(output_dir / "residual_plot.png", dpi=300)
plt.close(fig)

# ------------------------------------------
# GRAPH 4: FEATURE IMPORTANCE
# ------------------------------------------

fitted_preprocessor = best_model.named_steps["preprocessor"]
fitted_estimator = best_model.named_steps["model"]

if hasattr(fitted_estimator, "feature_importances_"):
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_estimator.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        "Importance", ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title(f"Top 15 Feature Importances - {best_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close(fig)

else:
    print(
        "\nFeature importance plot skipped: "
        "the selected model does not provide tree-based importances."
    )

print("\nAll available graphs saved in the outputs folder:")
for file in sorted(output_dir.glob("*.png")):
    print("-", file)

