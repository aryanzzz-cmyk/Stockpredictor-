# stock_prediction_BSE_PLUS_TOP5_COMPANIES_WITH_ENSEMBLE_SUMMARY.py
# -------------------------------------------------------------------------------
# FINAL INTEGRATED PROJECT FILE (WITH ENSEMBLE SUMMARY)
#
# PART 1: BSE SENSEX LSTM ENSEMBLE (RUNS FIRST)
# PART 2: TOP 5 BSE COMPANIES FROM DIFFERENT INDUSTRIES
#
# Industries Covered:
# ✔ Energy
# ✔ IT
# ✔ Banking
# ✔ Pharmaceuticals
# ✔ FMCG
#
# ✔ Ensemble summary printed for EACH asset
# ✔ Mean Accuracy, Std Accuracy, Ensemble Accuracy
# ✔ Academic & viva-ready output
# -------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ================= COMMON CONFIG =================
START_DATE = "2015-01-01"
END_DATE = "2021-01-01"
TRAIN_RATIO = 0.8
N_MODELS = 20
EPOCHS = 8
BATCH_SIZE = 16

# ================= MODEL =================
def build_lstm(seed, input_dim):
    tf.keras.utils.set_random_seed(seed)
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, input_dim)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

# ================= CORE PIPELINE =================
def run_prediction(ticker, name):
    print(f"\nRunning prediction for: {name}")

    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df.columns = df.columns.str.lower()

    df["return_1d"] = df["close"].pct_change()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["label"] = np.where(df["close"].shift(-3) > df["close"], 1, 0)
    df.dropna(inplace=True)

    # Volatility graph
    plt.figure(figsize=(10,5))
    plt.plot(df["date"], df["volatility_20"])
    plt.title(f"{name} – 20-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    split = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = train_df[["return_1d", "volatility_20"]]
    y_train = train_df["label"]
    X_test = test_df[["return_1d", "volatility_20"]]
    y_test = test_df["label"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    all_probs = []
    accuracies = []

    for i in range(N_MODELS):
        model = build_lstm(i, X_train.shape[2])
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        probs = model.predict(X_test).flatten()
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        all_probs.append(probs)

    ensemble_prob = np.mean(np.array(all_probs), axis=0)
    ensemble_pred = (ensemble_prob > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    # ================= ENSEMBLE SUMMARY =================
    print("ENSEMBLE SUMMARY")
    print("----------------")
    print(f"Mean Accuracy : {np.mean(accuracies):.4f}")
    print(f"Std Accuracy  : {np.std(accuracies):.4f}")
    print(f"Ensemble Acc  : {ensemble_acc:.4f}")

    results = pd.DataFrame({
        "Date": test_df["date"].values,
        "Confidence": ensemble_prob,
        "Prediction": ensemble_pred,
        "Actual": y_test.values
    })

    results["Correct"] = np.where(results["Prediction"] == results["Actual"], "✔", "✘")
    results.to_csv(name.replace(" ", "_") + "_LSTM_RESULTS.csv", index=False)

    # Prediction confidence graph
    plt.figure(figsize=(10,5))
    plt.plot(results["Date"], results["Confidence"])
    plt.title(f"{name} – LSTM Prediction Confidence")
    plt.xlabel("Date")
    plt.ylabel("Prediction Confidence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ================= PREDICTION (UP / DOWN) PLOT =================
    plt.figure(figsize=(10,4))
    plt.step(results["Date"], results["Prediction"], where="post")
    plt.yticks([0, 1], ["DOWN", "UP"])
    plt.title(f"{name} – LSTM Binary Prediction (UP / DOWN)")
    plt.xlabel("Date")
    plt.ylabel("Prediction")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ================= ACTUAL vs PREDICTION (DIRECTION) PLOT =================
    plt.figure(figsize=(10,5))
    plt.plot(results["Date"], results["Actual"], label="Actual (UP/DOWN)", linestyle="--")
    plt.plot(results["Date"], results["Prediction"], label="Predicted (UP/DOWN)", alpha=0.8)
    plt.yticks([0, 1], ["DOWN", "UP"])
    plt.title(f"{name} – Actual vs Predicted Direction")
    plt.xlabel("Date")
    plt.ylabel("Direction")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Correct vs Incorrect graph
    counts = results["Correct"].value_counts()
    plt.figure(figsize=(5,4))
    plt.bar("Correct", counts.get("✔", 0), color="green")
    plt.bar("Incorrect", counts.get("✘", 0), color="red")
    plt.title(f"{name} – Prediction Accuracy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# ================= PART 1: BSE SENSEX =================
run_prediction("^BSESN", "BSE Sensex")

# ================= PART 2: TOP 5 BSE COMPANIES =================
# Infosys replaced with SUN PHARMA (Pharmaceuticals)

COMPANIES = {
    "Reliance Industries (Energy)": "RELIANCE.NS",
    "TCS (IT Services)": "TCS.NS",
    "HDFC Bank (Banking)": "HDFCBANK.NS",
    "Sun Pharma (Pharmaceuticals)": "SUNPHARMA.NS",
    "ITC (FMCG)": "ITC.NS"
}

for name, ticker in COMPANIES.items():
    run_prediction(ticker, name)

print("\nScript finished successfully.")



# ================= SECTOR-WISE ACCURACY COMPARISON =================
sector_accuracy = []

def run_prediction_with_sector(ticker, name, sector):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.columns = df.columns.str.lower()
    df["return_1d"] = df["close"].pct_change()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["label"] = np.where(df["close"].shift(-3) > df["close"], 1, 0)
    df.dropna(inplace=True)

    split = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = train_df[["return_1d", "volatility_20"]]
    y_train = train_df["label"]
    X_test = test_df[["return_1d", "volatility_20"]]
    y_test = test_df["label"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    probs_all = []
    accs = []
    for i in range(N_MODELS):
        model = build_lstm(i, X_train.shape[2])
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        probs = model.predict(X_test).flatten()
        preds = (probs > 0.5).astype(int)
        accs.append(accuracy_score(y_test, preds))
        probs_all.append(probs)

    ensemble_prob = np.mean(np.array(probs_all), axis=0)
    ensemble_pred = (ensemble_prob > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    sector_accuracy.append({
        "Company": name,
        "Sector": sector,
        "Ensemble Accuracy": ensemble_acc
    })

# Run sector-wise tracking
for name, ticker in COMPANIES.items():
    sector = name.split("(")[1].replace(")", "")
    run_prediction_with_sector(ticker, name, sector)

# Create comparison table and graph
sector_df = pd.DataFrame(sector_accuracy)
print("\nSECTOR-WISE ENSEMBLE ACCURACY TABLE")
print(sector_df)

plt.figure(figsize=(10,5))
plt.bar(sector_df["Sector"], sector_df["Ensemble Accuracy"])
plt.title("Sector-wise LSTM Ensemble Accuracy Comparison")
plt.xlabel("Sector")
plt.ylabel("Accuracy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# ==============================================================================
# ===================== ADD-ON: PRICE PREDICTION MODULE =========================
# (Runs AFTER the original model, does NOT interfere with it)
# ==============================================================================

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

LOOKBACK = 20
EPOCHS_PRICE = 10

def run_price_prediction_addon(ticker, name):
    print(f"\n[ADD-ON] Running PRICE prediction for: {name}")

    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0)

    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    dates = df.index[-len(actual):]

    # ----------- Actual vs Predicted Price -----------
    plt.figure(figsize=(10,5))
    plt.plot(dates, actual, label="Actual Price")
    plt.plot(dates, preds, label="Predicted Price")
    plt.title(f"{name} – Actual vs Predicted Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ----------- Predicted Price Only -----------
    plt.figure(figsize=(10,5))
    plt.plot(dates, preds, label="Predicted Price", color="orange")
    plt.title(f"{name} – Predicted Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ----------- Metrics -----------
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)

    actual_dir = np.sign(np.diff(actual.flatten()))
    pred_dir = np.sign(np.diff(preds.flatten()))
    directional_accuracy = np.mean(actual_dir == pred_dir)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")

    # ----------- Prediction Error Plot -----------
    plt.figure(figsize=(10,4))
    plt.plot(dates, actual.flatten() - preds.flatten())
    plt.axhline(0, linestyle='--', alpha=0.5)
    plt.title(f"{name} – Prediction Error (Actual − Predicted)")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ================= RUN PRICE ADD-ONS =================
print("\n================ RUNNING PRICE PREDICTION ADD-ONS ================")

run_price_prediction_addon("^BSESN", "BSE Sensex")

for name, ticker in COMPANIES.items():
    run_price_prediction_addon(ticker, name)

print("\nAll add-on price predictions completed.")



# ==============================================================================
# ================= ADD-ON: ROLLING RMSE & DASHBOARD ============================
# ==============================================================================

def rolling_rmse(actual, predicted, window=20):
    rmses = []
    for i in range(len(actual)):
        if i < window:
            rmses.append(np.nan)
        else:
            rmses.append(
                np.sqrt(mean_squared_error(actual[i-window:i], predicted[i-window:i]))
            )
    return np.array(rmses)


def run_dashboard_addon(ticker, name):
    print(f"\n[ADD-ON DASHBOARD] {name}")

    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0)

    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    dates = df.index[-len(actual):]

    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)

    actual_dir = np.sign(np.diff(actual.flatten()))
    pred_dir = np.sign(np.diff(preds.flatten()))
    directional_accuracy = np.mean(actual_dir == pred_dir)

    roll_rmse = rolling_rmse(actual.flatten(), preds.flatten(), window=20)

    # ================= DASHBOARD =================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} – LSTM Price Prediction Dashboard", fontsize=14)

    # (1) Actual vs Predicted
    axs[0, 0].plot(dates, actual, label="Actual")
    axs[0, 0].plot(dates, preds, label="Predicted")
    axs[0, 0].set_title("Actual vs Predicted Price")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)

    # (2) Predicted Price
    axs[0, 1].plot(dates, preds, color="orange")
    axs[0, 1].set_title("Predicted Price")
    axs[0, 1].grid(alpha=0.3)

    # (3) Prediction Error
    axs[1, 0].plot(dates, actual.flatten() - preds.flatten())
    axs[1, 0].axhline(0, linestyle="--", alpha=0.5)
    axs[1, 0].set_title("Prediction Error")
    axs[1, 0].grid(alpha=0.3)

    # (4) Rolling RMSE
    axs[1, 1].plot(dates, roll_rmse, color="red")
    axs[1, 1].set_title("Rolling RMSE (Window=20)")
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | Directional Accuracy: {directional_accuracy:.2%}")


print("\n================ RUNNING DASHBOARD ADD-ONS ================")

run_dashboard_addon("^BSESN", "BSE Sensex")

for name, ticker in COMPANIES.items():
    run_dashboard_addon(ticker, name)

print("\nAll dashboard add-ons completed.")



# ==============================================================================
# ================= ADD-ON: MULTI-STEP FUTURE FORECASTING =======================
# ==============================================================================

FORECAST_HORIZON = 30  # days into the future

def run_future_forecast_addon(ticker, name):
    print(f"\n[FUTURE FORECAST] {name} ({FORECAST_HORIZON} days)")

    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0)

    last_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    future_scaled = []

    for _ in range(FORECAST_HORIZON):
        next_pred = model.predict(last_window, verbose=0)[0, 0]
        future_scaled.append(next_pred)
        last_window = np.roll(last_window, -1, axis=1)
        last_window[0, -1, 0] = next_pred

    future_prices = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    )

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="B"
    )

    # ================= FUTURE FORECAST PLOT =================
    plt.figure(figsize=(10,5))
    plt.plot(df.index[-100:], df['Close'].values[-100:], label="Recent Actual")
    plt.plot(future_dates, future_prices, label="Future Forecast", color="red")
    plt.title(f"{name} – {FORECAST_HORIZON}-Day Future Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return future_prices


print("\n================ RUNNING MULTI-STEP FUTURE FORECASTS ================")

run_future_forecast_addon("^BSESN", "BSE Sensex")

for name, ticker in COMPANIES.items():
    run_future_forecast_addon(ticker, name)

print("\nAll future forecasts completed.")



# ==============================================================================
# ============== ADD-ON: SECTOR-LEVEL FUTURE TREND COMPARISON ====================
# ==============================================================================

SECTOR_MAP = {
    "Reliance Industries (Energy)": "Energy",
    "TCS (IT Services)": "IT",
    "HDFC Bank (Banking)": "Banking",
    "Sun Pharma (Pharmaceuticals)": "Pharmaceuticals",
    "ITC (FMCG)": "FMCG"
}

def sector_future_trend_forecast():
    print("\n================ SECTOR-LEVEL FUTURE TREND ANALYSIS ================")

    sector_forecasts = {}

    for name, ticker in COMPANIES.items():
        sector = SECTOR_MAP[name]

        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
        df = df[['Close']].dropna()

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i, 0])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0)

        last_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        future_scaled = []

        for _ in range(FORECAST_HORIZON):
            next_pred = model.predict(last_window, verbose=0)[0, 0]
            future_scaled.append(next_pred)
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1, 0] = next_pred

        future_prices = scaler.inverse_transform(
            np.array(future_scaled).reshape(-1, 1)
        ).flatten()

        sector_forecasts.setdefault(sector, []).append(future_prices)

    # Average future trend per sector
    plt.figure(figsize=(12,6))

    for sector, forecasts in sector_forecasts.items():
        avg_trend = np.mean(np.array(forecasts), axis=0)
        plt.plot(avg_trend, label=f"{sector} Sector")

    plt.title("Sector-Level Average Future Price Trend Comparison")
    plt.xlabel("Future Days")
    plt.ylabel("Price (Relative Scale)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Sector-level future trend comparison completed.")

sector_future_trend_forecast()



# ==============================================================================
# ============== ADD-ON: SECTOR vs INDEX FUTURE TREND COMPARISON =================
# ==============================================================================

def sector_vs_index_future_comparison():
    print("\n================ SECTOR vs INDEX FUTURE COMPARISON ================")

    # ---------- Forecast INDEX (BSE Sensex) ----------
    df_index = yf.download("^BSESN", start=START_DATE, end=END_DATE, auto_adjust=True)
    df_index = df_index[['Close']].dropna()

    scaler_index = MinMaxScaler()
    scaled_index = scaler_index.fit_transform(df_index)

    X_idx, y_idx = [], []
    for i in range(LOOKBACK, len(scaled_index)):
        X_idx.append(scaled_index[i-LOOKBACK:i, 0])
        y_idx.append(scaled_index[i, 0])

    X_idx, y_idx = np.array(X_idx), np.array(y_idx)
    X_idx = X_idx.reshape((X_idx.shape[0], X_idx.shape[1], 1))

    model_index = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model_index.compile(optimizer="adam", loss="mse")
    model_index.fit(X_idx, y_idx, epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0)

    last_idx = scaled_index[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    future_idx_scaled = []

    for _ in range(FORECAST_HORIZON):
        p = model_index.predict(last_idx, verbose=0)[0, 0]
        future_idx_scaled.append(p)
        last_idx = np.roll(last_idx, -1, axis=1)
        last_idx[0, -1, 0] = p

    future_index = scaler_index.inverse_transform(
        np.array(future_idx_scaled).reshape(-1, 1)
    ).flatten()

    # ---------- Sector Averages (already learned structure) ----------
    sector_forecasts = {}

    for name, ticker in COMPANIES.items():
        sector = SECTOR_MAP[name]

        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
        df = df[['Close']].dropna()

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i, 0])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0)

        last_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        future_scaled = []

        for _ in range(FORECAST_HORIZON):
            p = model.predict(last_window, verbose=0)[0, 0]
            future_scaled.append(p)
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1, 0] = p

        future_prices = scaler.inverse_transform(
            np.array(future_scaled).reshape(-1, 1)
        ).flatten()

        sector_forecasts.setdefault(sector, []).append(future_prices)

    # ---------- Plot Comparison ----------
    plt.figure(figsize=(12,6))

    plt.plot(future_index, label="BSE Sensex (Index)", linewidth=3, color="black")

    for sector, forecasts in sector_forecasts.items():
        avg_trend = np.mean(np.array(forecasts), axis=0)
        plt.plot(avg_trend, label=f"{sector} Sector")

    plt.title("Sector vs Index – Future Trend Comparison")
    plt.xlabel("Future Days")
    plt.ylabel("Price (Relative Scale)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Sector vs Index future comparison completed.")


sector_vs_index_future_comparison()



# ==============================================================================
# ============== ADD-ON: RELATIVE STRENGTH (SECTOR ÷ INDEX) RATIO ================
# ==============================================================================

def sector_relative_strength_ratio():
    print("\n================ SECTOR RELATIVE STRENGTH ANALYSIS ================")

    # -------- Forecast INDEX (Sensex) --------
    df_index = yf.download("^BSESN", start=START_DATE, end=END_DATE, auto_adjust=True)
    df_index = df_index[['Close']].dropna()

    scaler_index = MinMaxScaler()
    scaled_index = scaler_index.fit_transform(df_index)

    X_idx = []
    for i in range(LOOKBACK, len(scaled_index)):
        X_idx.append(scaled_index[i-LOOKBACK:i, 0])
    X_idx = np.array(X_idx).reshape(-1, LOOKBACK, 1)

    model_index = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model_index.compile(optimizer="adam", loss="mse")
    model_index.fit(X_idx, scaled_index[LOOKBACK:], epochs=EPOCHS_PRICE,
                    batch_size=BATCH_SIZE, verbose=0)

    last_idx = scaled_index[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    future_idx_scaled = []

    for _ in range(FORECAST_HORIZON):
        p = model_index.predict(last_idx, verbose=0)[0, 0]
        future_idx_scaled.append(p)
        last_idx = np.roll(last_idx, -1, axis=1)
        last_idx[0, -1, 0] = p

    future_index = scaler_index.inverse_transform(
        np.array(future_idx_scaled).reshape(-1, 1)
    ).flatten()

    # -------- Sector Relative Strength --------
    plt.figure(figsize=(12,6))

    for name, ticker in COMPANIES.items():
        sector = SECTOR_MAP[name]

        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
        df = df[['Close']].dropna()

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        X = []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i, 0])
        X = np.array(X).reshape(-1, LOOKBACK, 1)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, scaled[LOOKBACK:], epochs=EPOCHS_PRICE,
                  batch_size=BATCH_SIZE, verbose=0)

        last_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        future_scaled = []

        for _ in range(FORECAST_HORIZON):
            p = model.predict(last_window, verbose=0)[0, 0]
            future_scaled.append(p)
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1, 0] = p

        future_sector = scaler.inverse_transform(
            np.array(future_scaled).reshape(-1, 1)
        ).flatten()

        rs_ratio = future_sector / future_index
        plt.plot(rs_ratio, label=f"{sector} RS Ratio")

    plt.axhline(1.0, linestyle="--", color="black", alpha=0.6)
    plt.title("Relative Strength (Sector ÷ Index) – Future Trend")
    plt.xlabel("Future Days")
    plt.ylabel("Relative Strength Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Relative strength analysis completed.")


sector_relative_strength_ratio()



# ==============================================================================
# ============== ADD-ON: RELATIVE STRENGTH HEATMAP ==============================
# ==============================================================================

def sector_relative_strength_heatmap():
    print("\n================ RELATIVE STRENGTH HEATMAP ================")

    # --- Forecast Index once ---
    df_index = yf.download("^BSESN", start=START_DATE, end=END_DATE, auto_adjust=True)
    df_index = df_index[['Close']].dropna()

    scaler_index = MinMaxScaler()
    scaled_index = scaler_index.fit_transform(df_index)

    X_idx = []
    for i in range(LOOKBACK, len(scaled_index)):
        X_idx.append(scaled_index[i-LOOKBACK:i, 0])
    X_idx = np.array(X_idx).reshape(-1, LOOKBACK, 1)

    model_index = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model_index.compile(optimizer="adam", loss="mse")
    model_index.fit(
        X_idx, scaled_index[LOOKBACK:],
        epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0
    )

    last_idx = scaled_index[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    future_idx_scaled = []

    for _ in range(FORECAST_HORIZON):
        p = model_index.predict(last_idx, verbose=0)[0, 0]
        future_idx_scaled.append(p)
        last_idx = np.roll(last_idx, -1, axis=1)
        last_idx[0, -1, 0] = p

    future_index = scaler_index.inverse_transform(
        np.array(future_idx_scaled).reshape(-1, 1)
    ).flatten()

    # --- Sector RS matrix ---
    rs_matrix = {}

    for name, ticker in COMPANIES.items():
        sector = SECTOR_MAP[name]

        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
        df = df[['Close']].dropna()

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        X = []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i, 0])
        X = np.array(X).reshape(-1, LOOKBACK, 1)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X, scaled[LOOKBACK:],
            epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0
        )

        last_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        future_scaled = []

        for _ in range(FORECAST_HORIZON):
            p = model.predict(last_window, verbose=0)[0, 0]
            future_scaled.append(p)
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1, 0] = p

        future_sector = scaler.inverse_transform(
            np.array(future_scaled).reshape(-1, 1)
        ).flatten()

        rs_matrix[sector] = future_sector / future_index

    rs_df = pd.DataFrame(rs_matrix)

    # --- Heatmap plot ---
    plt.figure(figsize=(12,6))
    plt.imshow(rs_df.T, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Relative Strength (Sector ÷ Index)")
    plt.yticks(range(len(rs_df.columns)), rs_df.columns)
    plt.xticks(
        ticks=np.linspace(0, FORECAST_HORIZON - 1, 6, dtype=int),
        labels=np.linspace(1, FORECAST_HORIZON, 6, dtype=int)
    )
    plt.xlabel("Future Days")
    plt.ylabel("Sector")
    plt.title("Relative Strength Heatmap (Sector vs Index)")
    plt.tight_layout()
    plt.show()

    print("Relative strength heatmap completed.")


sector_relative_strength_heatmap()



# ==============================================================================
# ============== ADD-ON: MONTE CARLO DROPOUT CONFIDENCE BANDS ====================
# ==============================================================================

MC_SAMPLES = 50

def mc_dropout_forecast(ticker, name):
    print(f"\n[MC DROPOUT] {name}")

    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, LOOKBACK, 1)
    y = np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=EPOCHS_PRICE, batch_size=BATCH_SIZE, verbose=0)

    last_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)

    forecasts = []
    for _ in range(MC_SAMPLES):
        preds = []
        window = last_window.copy()
        for _ in range(FORECAST_HORIZON):
            p = model(window, training=True).numpy()[0, 0]
            preds.append(p)
            window = np.roll(window, -1, axis=1)
            window[0, -1, 0] = p
        forecasts.append(preds)

    forecasts = np.array(forecasts)
    mean_forecast = forecasts.mean(axis=0)
    std_forecast = forecasts.std(axis=0)

    mean_price = scaler.inverse_transform(mean_forecast.reshape(-1, 1)).flatten()
    upper = scaler.inverse_transform((mean_forecast + 2 * std_forecast).reshape(-1, 1)).flatten()
    lower = scaler.inverse_transform((mean_forecast - 2 * std_forecast).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="B"
    )

    plt.figure(figsize=(10,5))
    plt.plot(future_dates, mean_price, label="Mean Forecast")
    plt.fill_between(future_dates, lower, upper, alpha=0.3, label="±2σ Confidence Band")
    plt.title(f"{name} – Monte Carlo Dropout Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


print("\n================ RUNNING MONTE CARLO DROPOUT FORECASTS ================")

mc_dropout_forecast("^BSESN", "BSE Sensex")

for name, ticker in COMPANIES.items():
    mc_dropout_forecast(ticker, name)

print("\nMonte Carlo Dropout analysis completed.")
