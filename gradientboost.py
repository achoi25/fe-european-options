import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = "clean_data_2/calls/"
START_YEAR = 2010
END_YEAR = 2023

FEATURE_COLS = [
    'UNDERLYING_LAST', 'STRIKE', 'DTE', 'C_BID', 'C_ASK', 'C_LAST',
    'C_VOLUME', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO',
    'MidPrice_Call', 'Bid_Ask_Spread_Call', 'TimeToExpiryYears'
]
TARGET_COL = 'C_IV'

results = []

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        ym_str = f"{year}{month:02d}"
        filename = f"calls_spx_eod_{ym_str}.pkl"
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_pickle(filepath)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        if not all(col in df.columns for col in FEATURE_COLS + [TARGET_COL]):
            continue

        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
        if df.empty:
            continue

        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'year_month': ym_str,
            'rows': len(df),
            'mse': mse,
            'r2': r2
        })

        print(f"{filename}: MSE={mse:.5f}, R²={r2:.5f}")

        # === Visualization for each month ===
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.xlabel("Actual C_IV")
        plt.ylabel("Predicted C_IV")
        plt.title(f"Predicted vs Actual C_IV — {ym_str}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, f"pred_vs_actual_{ym_str}.png"))
        plt.close()

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(8, 5))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel("Feature Importance")
        plt.title(f"Feature Importances — {ym_str}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, f"feature_importance_{ym_str}.png"))
        plt.close()

summary_df = pd.DataFrame(results)
summary_path = os.path.join(DATA_DIR, "gradientboost_results.csv")
summary_df.to_csv(summary_path, index=False)

plt.figure(figsize=(10, 6))
plt.plot(summary_df['year_month'], summary_df['mse'], marker='o', label='MSE')
plt.plot(summary_df['year_month'], summary_df['r2'], marker='s', label='R²')
plt.xlabel("Year-Month")
plt.ylabel("Metric Value")
plt.title("Model Performance Over Time")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "performance_over_time.png"))
plt.close()

print("All files processed. Results and plots saved to:", DATA_DIR)
