"""
Ejercicio 1: modelos LSTM con tuneo de parametros para dos series de tiempo,
y comparacion contra los mejores modelos clasicos del laboratorio anterior.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from datos_comunes import cargar_series

tf.random.set_seed(42)
np.random.seed(42)

OUT_DIR = "resultados"
VENTANA = 12
SERIES_ELEGIDAS = ["total", "via_aerea"]


def crear_ventanas(arr, ventana):
    x, y = [], []
    for i in range(len(arr) - ventana):
        x.append(arr[i:i + ventana])
        y.append(arr[i + ventana])
    return np.array(x), np.array(y)


def construir_modelo(tipo, units, lr=0.001, dropout=0.0):
    modelo = Sequential()
    modelo.add(Input(shape=(VENTANA, 1)))
    if tipo == "simple":
        modelo.add(LSTM(units))
    else:  # "dropout"
        modelo.add(LSTM(units))
        modelo.add(Dropout(dropout))
    modelo.add(Dense(1))
    modelo.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return modelo


def evaluar(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def tunear_serie(nombre, train, test):
    print(f"\n{'='*70}\nSERIE: {nombre}  (train={len(train)}, test={len(test)})\n{'='*70}")

    scaler = MinMaxScaler()
    train_esc = scaler.fit_transform(train.values.reshape(-1, 1))
    test_esc = scaler.transform(test.values.reshape(-1, 1))

    x_all, y_all = crear_ventanas(train_esc, VENTANA)
    n_val = max(12, int(len(x_all) * 0.2))
    x_tr, y_tr = x_all[:-n_val], y_all[:-n_val]
    x_val, y_val = x_all[-n_val:], y_all[-n_val:]

    serie_para_test = np.concatenate([train_esc[-VENTANA:], test_esc])
    x_test, y_test = crear_ventanas(serie_para_test, VENTANA)

    grid = []
    for units in (32, 64):
        for lr in (0.001, 0.01):
            grid.append(("Modelo 1 (LSTM simple)", "simple", units, lr, 0.0))
    for units in (32, 64):
        for dropout in (0.1, 0.3):
            grid.append(("Modelo 2 (LSTM + Dropout)", "dropout", units, 0.001, dropout))

    registros = []
    mejores = {}  
    for etiqueta, tipo, units, lr, dropout in grid:
        modelo = construir_modelo(tipo, units, lr, dropout)
        es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        modelo.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                   epochs=150, batch_size=16, verbose=0, callbacks=[es])
        val_pred = modelo.predict(x_val, verbose=0)
        val_mae, val_rmse = evaluar(y_val, val_pred)

        reg = {"serie": nombre, "familia": etiqueta, "units": units,
               "learning_rate": lr, "dropout": dropout, "val_MAE": val_mae,
               "val_RMSE": val_rmse}
        registros.append(reg)
        print(f"  {etiqueta:28s} units={units:3d} lr={lr:<6} dropout={dropout:<4} "
              f"-> val_RMSE(escala 0-1)={val_rmse:.4f}")

        if etiqueta not in mejores or val_rmse < mejores[etiqueta]["val_RMSE"]:
            mejores[etiqueta] = reg

    tabla_grid = pd.DataFrame(registros)

    finales = []
    for etiqueta, reg in mejores.items():
        tipo = "simple" if "simple" in etiqueta else "dropout"
        modelo = construir_modelo(tipo, reg["units"], reg["learning_rate"], reg["dropout"])
        es = EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
        modelo.fit(x_all, y_all, epochs=150, batch_size=16, verbose=0, callbacks=[es])

        pred_esc = modelo.predict(x_test, verbose=0)
        pred = scaler.inverse_transform(pred_esc).flatten()
        real = scaler.inverse_transform(y_test).flatten()
        mae, rmse = evaluar(real, pred)

        finales.append({**reg, "test_MAE": mae, "test_RMSE": rmse,
                         "pred": pred, "real": real, "modelo_obj": modelo})
        print(f"  --> {etiqueta} finalista: test MAE={mae:,.0f}  RMSE={rmse:,.0f}")

    mejor_final = min(finales, key=lambda r: r["test_RMSE"])
    print(f"\n  ### Mejor modelo LSTM para '{nombre}': {mejor_final['familia']} "
          f"(units={mejor_final['units']}, lr={mejor_final['learning_rate']}, "
          f"dropout={mejor_final['dropout']}) "
          f"-> test MAE={mejor_final['test_MAE']:,.0f}  RMSE={mejor_final['test_RMSE']:,.0f}")

    # Grafico real vs prediccion del mejor modelo
    fig, ax = plt.subplots(figsize=(11, 4.5))
    idx = test.index[-len(mejor_final["real"]):]
    ax.plot(idx, mejor_final["real"], label="Real", color="black", lw=2)
    for f in finales:
        ax.plot(idx, f["pred"], "--", label=f"{f['familia']} (pred)")
    ax.set_title(f"Prediccion en test - {nombre}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ej1_pred_{nombre}.png", dpi=120)
    plt.close()

    return tabla_grid, finales, mejor_final, scaler


def clasicos_benchmark(nombre, train, test):
    """Recalcula los mejores modelos clasicos (lab1) sobre el MISMO
    train/test usado aqui, para que la comparacion del punto 1.4 sea
    consistente."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from pmdarima import auto_arima
    from prophet import Prophet
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    horizon = len(test)

    def log_t(s):
        return np.log1p(s) if (s <= 0).any() else np.log(s)

    def inv_log_t(s_log, original):
        return np.expm1(s_log) if (original <= 0).any() else np.exp(s_log)

    resultados = []

    # ARIMA
    log_train = log_t(train)
    auto = auto_arima(log_train, seasonal=True, m=12, max_p=3, max_q=3,
                       max_P=1, max_Q=1, stepwise=True, suppress_warnings=True,
                       error_action="ignore")
    fit = SARIMAX(log_train, order=auto.order, seasonal_order=auto.seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred_log = fit.get_forecast(steps=horizon).predicted_mean
    pred_arima = inv_log_t(pred_log, train).clip(lower=0)
    mae, rmse = evaluar(test.values, pred_arima.values)
    resultados.append({"modelo": f"ARIMA{auto.order}x{auto.seasonal_order}", "MAE": mae, "RMSE": rmse})

    # Prophet
    df_p = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    fut = pd.DataFrame({"ds": test.index})
    pred_prophet = m.predict(fut)["yhat"].clip(lower=0).values
    mae, rmse = evaluar(test.values, pred_prophet)
    resultados.append({"modelo": "Prophet", "MAE": mae, "RMSE": rmse})

    # Holt-Winters
    seasonal_type = "mul" if (train > 0).all() else "add"
    hw = ExponentialSmoothing(train, trend="add", seasonal=seasonal_type, seasonal_periods=12).fit()
    pred_hw = hw.forecast(horizon).clip(lower=0)
    mae, rmse = evaluar(test.values, pred_hw.values)
    resultados.append({"modelo": "Holt-Winters", "MAE": mae, "RMSE": rmse})

    # Suavizamiento exponencial simple
    ses = SimpleExpSmoothing(train).fit()
    pred_ses = ses.forecast(horizon).clip(lower=0)
    mae, rmse = evaluar(test.values, pred_ses.values)
    resultados.append({"modelo": "Suavizamiento exponencial simple", "MAE": mae, "RMSE": rmse})

    tabla = pd.DataFrame(resultados).sort_values("RMSE")
    tabla.insert(0, "serie", nombre)
    print(f"\n  Clasicos ({nombre}):")
    print(tabla.to_string(index=False))
    return tabla


def main():
    series_train, series_test, labels = cargar_series()

    todas_grid = []
    todas_finales = []
    todas_clasicas = []

    for nombre in SERIES_ELEGIDAS:
        train, test = series_train[nombre], series_test[nombre]
        tabla_grid, finales, mejor_final, scaler = tunear_serie(nombre, train, test)
        todas_grid.append(tabla_grid)

        for f in finales:
            todas_finales.append({
                "serie": nombre, "familia": f["familia"], "units": f["units"],
                "learning_rate": f["learning_rate"], "dropout": f["dropout"],
                "test_MAE": f["test_MAE"], "test_RMSE": f["test_RMSE"],
                "es_mejor": f is mejor_final,
            })

        tabla_clasicos = clasicos_benchmark(nombre, train, test)
        todas_clasicas.append(tabla_clasicos)

    pd.concat(todas_grid, ignore_index=True).to_csv(f"{OUT_DIR}/ej1_grid.csv", index=False)
    pd.DataFrame(todas_finales).to_csv(f"{OUT_DIR}/ej1_finales.csv", index=False)
    pd.concat(todas_clasicas, ignore_index=True).to_csv(f"{OUT_DIR}/ej1_clasicos.csv", index=False)
    print("\nListo. Resultados guardados en resultados/ej1_*.csv y ej1_pred_*.png")


if __name__ == "__main__":
    main()
