"""
Punto 2.14: un modelo de LSTM que en vez de recibir los valores crudos de
la ventana recibe las 22 caracteristicas de catch22 calculadas sobre esa ventana 
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pycatch22
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from datos_comunes import cargar_series

tf.random.set_seed(42)
np.random.seed(42)

OUT_DIR = "resultados"
SERIE = "total"
VENTANA = 12   
LOOKBACK = 3   


def evaluar(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def main():
    series_train, series_test, labels = cargar_series()
    train, test = series_train[SERIE], series_test[SERIE]
    completa = pd.concat([train, test]).sort_index()
    n_train = len(train)

    valores = completa.values.astype(float)
    feats, targets, idx_target = [], [], []
    for t in range(VENTANA, len(valores) - 1):
        ventana_vals = valores[t - VENTANA:t]
        res = pycatch22.catch22_all(list(ventana_vals))
        feats.append(res["values"])
        targets.append(valores[t + 1])
        idx_target.append(t + 1)
    nombres_feat = res["names"]
    feats = np.array(feats)
    targets = np.array(targets)
    idx_target = np.array(idx_target)

    X, y, idx_y = [], [], []
    for i in range(LOOKBACK, len(feats)):
        X.append(feats[i - LOOKBACK:i])
        y.append(targets[i])
        idx_y.append(idx_target[i])
    X = np.array(X)
    y = np.array(y)
    idx_y = np.array(idx_y)

    es_train = idx_y < n_train
    X_train, y_train = X[es_train], y[es_train]
    X_test, y_test = X[~es_train], y[~es_train]
    fechas_test = completa.index[idx_y[~es_train]]

    n_feat = X_train.shape[2]
    scaler_x = StandardScaler().fit(X_train.reshape(-1, n_feat))
    X_train_s = scaler_x.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test_s = scaler_x.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    scaler_y = MinMaxScaler().fit(y_train.reshape(-1, 1))
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1))

    n_val = max(10, int(len(X_train_s) * 0.2))
    x_tr, y_tr = X_train_s[:-n_val], y_train_s[:-n_val]
    x_val, y_val = X_train_s[-n_val:], y_train_s[-n_val:]

    modelo = Sequential([
        Input(shape=(LOOKBACK, n_feat)),
        LSTM(32),
        Dense(1),
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    modelo.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=150,
               batch_size=16, verbose=0, callbacks=[es])

    pred_s = modelo.predict(X_test_s, verbose=0)
    pred = scaler_y.inverse_transform(pred_s).flatten()
    mae, rmse = evaluar(y_test, pred)
    print(f"LSTM + catch22 rolling (serie={SERIE}): test MAE={mae:,.0f}  RMSE={rmse:,.0f}")

    # --- 6. comparacion contra el mejor LSTM "clasico" (ej1) --------------
    finales = pd.read_csv(f"{OUT_DIR}/ej1_finales.csv")
    fila_mejor = finales[(finales["serie"] == SERIE) & (finales["es_mejor"])].iloc[0]
    print(f"Mejor LSTM clasico (ej1, serie={SERIE}): "
          f"MAE={fila_mejor['test_MAE']:,.0f}  RMSE={fila_mejor['test_RMSE']:,.0f}")

    resumen = pd.DataFrame([
        {"modelo": "LSTM (ventana de valores crudos) - mejor de Ejercicio 1",
         "MAE": fila_mejor["test_MAE"], "RMSE": fila_mejor["test_RMSE"]},
        {"modelo": "LSTM + caracteristicas catch22 (rolling)", "MAE": mae, "RMSE": rmse},
    ])
    resumen.to_csv(f"{OUT_DIR}/ej2_lstm_catch22_comparacion.csv", index=False)
    print("\n" + resumen.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(fechas_test, y_test, label="Real", color="black", lw=2)
    ax.plot(fechas_test, pred, "--", label="LSTM + catch22 (pred)")
    ax.set_title(f"LSTM con caracteristicas catch22 - {SERIE}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ej2_lstm_catch22_pred.png", dpi=120)
    plt.close()

    print("\nListo. Resultados guardados en resultados/ej2_lstm_catch22_*")


if __name__ == "__main__":
    main()
