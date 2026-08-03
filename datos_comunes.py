"""
Construccion de series y particion train/test.
"""
import pandas as pd
import numpy as np

RAW_PATH = "datos/Laboratorio 2. Series de Tiempo 2026 - Base_Migracion_2009-2026jun.xlsx"


def _construir_serie(df_split, col=None, valor=None, meses_ref=None):
    if col is not None:
        df_split = df_split[df_split[col] == valor]
    serie = df_split.groupby("Fecha")["Viajero"].sum().sort_index()
    if meses_ref is not None:
        serie = serie.reindex(meses_ref, fill_value=0)
    serie.index.freq = "MS"
    return serie


def cargar_series():
    """Devuelve (series_train, series_test, labels): dicts nombre -> pd.Series."""
    df = pd.read_excel(RAW_PATH, sheet_name="Datos")
    df.columns = ["Anio", "MesCod", "Mes", "Via", "Frontera", "Pais", "Region",
                  "RegionDos", "RegionOMT", "MCEO", "AgrupResidencia",
                  "TipoViajero", "Viajero"]
    df["Fecha"] = pd.to_datetime(dict(year=df["Anio"], month=df["MesCod"], day=1))

    # Solo Turista + Excursionista: categorias comparables en todo 2009-2026
    df_comp = df[df["TipoViajero"].isin(["Turista", "Excursionista"])].copy()

    meses = np.sort(df_comp["Fecha"].unique())
    n_train = int(round(len(meses) * 0.7))
    train_meses, test_meses = meses[:n_train], meses[n_train:]

    df_train = df_comp[df_comp["Fecha"].isin(train_meses)]
    df_test = df_comp[df_comp["Fecha"].isin(test_meses)]

    top_regiones = (
        df_comp.groupby("RegionDos")["Viajero"].sum()
        .sort_values(ascending=False).head(3)
    )
    reg1, reg2, reg3 = top_regiones.index

    definiciones = {
        "total": (None, None),
        "via_aerea": ("Via", "Aérea"),
        "via_terrestre": ("Via", "Terrestre"),
        "via_maritima": ("Via", "Marítima"),
        "region1": ("RegionDos", reg1),
        "region2": ("RegionDos", reg2),
        "region3": ("RegionDos", reg3),
    }
    labels = {
        "total": "Total mensual de viajeros (Turista+Excursionista)",
        "via_aerea": "Viajeros por vía Aérea",
        "via_terrestre": "Viajeros por vía Terrestre",
        "via_maritima": "Viajeros por vía Marítima",
        "region1": f"Viajeros por región: {reg1}",
        "region2": f"Viajeros por región: {reg2}",
        "region3": f"Viajeros por región: {reg3}",
    }

    series_train, series_test = {}, {}
    for nombre, (col, val) in definiciones.items():
        series_train[nombre] = _construir_serie(df_train, col, val, train_meses)
        series_test[nombre] = _construir_serie(df_test, col, val, test_meses)

    return series_train, series_test, labels


def serie_completa(nombre):
    """Serie completa (train+test concatenado) para un nombre dado."""
    series_train, series_test, _ = cargar_series()
    completa = pd.concat([series_train[nombre], series_test[nombre]]).sort_index()
    completa.index.freq = "MS"
    return completa


if __name__ == "__main__":
    series_train, series_test, labels = cargar_series()
    for nombre in labels:
        print(f"{nombre:15s} train={len(series_train[nombre]):3d} "
              f"test={len(series_test[nombre]):3d}  {labels[nombre]}")
