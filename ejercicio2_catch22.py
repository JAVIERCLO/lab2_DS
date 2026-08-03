"""
Ejercicio 2: caracterizacion de las series con catch22.
Usa las 7 series completas: total, via_aerea, via_terrestre, via_maritima, region1, region2, region3.

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.tsa.seasonal import seasonal_decompose

from datos_comunes import cargar_series

OUT_DIR = "resultados"
np.random.seed(42)


def main():
    series_train, series_test, labels = cargar_series()
    nombres = list(labels.keys())
    completas = {n: pd.concat([series_train[n], series_test[n]]).sort_index() for n in nombres}

    # -----------------------------------------------------------------
    # 2.2 / 2.3 Extraccion de las 22 caracteristicas de catch22
    # -----------------------------------------------------------------
    filas = {}
    nombres_features = None
    for n in nombres:
        res = pycatch22.catch22_all(list(completas[n].astype(float).values))
        filas[n] = res["values"]
        nombres_features = res["names"]

    matriz = pd.DataFrame(filas, index=nombres_features).T  # filas=series, cols=features
    matriz.index.name = "serie"
    matriz.to_csv(f"{OUT_DIR}/ej2_features.csv")
    print("Matriz de caracteristicas (series x 22):")
    print(matriz.round(3).to_string())

    # -----------------------------------------------------------------
    # 2.4 Estandarizacion
    # -----------------------------------------------------------------
    scaler = StandardScaler()
    matriz_std = pd.DataFrame(scaler.fit_transform(matriz), index=matriz.index, columns=matriz.columns)
    matriz_std.to_csv(f"{OUT_DIR}/ej2_features_std.csv")

    # -----------------------------------------------------------------
    # 2.5 PCA
    # -----------------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matriz_std)
    df_pca = pd.DataFrame(coords, index=matriz.index, columns=["PC1", "PC2"])
    df_pca["var_explicada_PC1"] = pca.explained_variance_ratio_[0]
    df_pca["var_explicada_PC2"] = pca.explained_variance_ratio_[1]
    df_pca.to_csv(f"{OUT_DIR}/ej2_pca.csv")
    print(f"\nVarianza explicada: PC1={pca.explained_variance_ratio_[0]:.1%}  "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")

    loadings = pd.DataFrame(pca.components_.T, index=matriz.columns, columns=["PC1", "PC2"])
    loadings.to_csv(f"{OUT_DIR}/ej2_pca_loadings.csv")
    print("\nCaracteristicas con mayor peso en PC1:")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(5))
    print("\nCaracteristicas con mayor peso en PC2:")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(5))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df_pca["PC1"], df_pca["PC2"], s=90, color="steelblue")
    for n in matriz.index:
        ax.annotate(n, (df_pca.loc[n, "PC1"], df_pca.loc[n, "PC2"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA de caracteristicas catch22 por serie")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ej2_pca.png", dpi=120)
    plt.close()

    # -----------------------------------------------------------------
    # Clustering: jerarquico (dendrograma) + KMeans
    # -----------------------------------------------------------------
    Z = linkage(matriz_std.values, method="ward")
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=list(matriz.index), ax=ax)
    ax.set_title("Dendrograma (distancia euclidiana, enlace Ward)")
    ax.set_ylabel("Distancia")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ej2_dendrograma.png", dpi=120)
    plt.close()

    mejor_k, mejor_sil = 2, -1
    for k in range(2, min(5, len(matriz) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(matriz_std)
        sil = silhouette_score(matriz_std, km.labels_)
        print(f"KMeans k={k}  silhouette={sil:.3f}")
        if sil > mejor_sil:
            mejor_k, mejor_sil = k, sil

    km = KMeans(n_clusters=mejor_k, random_state=42, n_init=10).fit(matriz_std)
    df_clusters = pd.DataFrame({"serie": matriz.index, "cluster": km.labels_})
    df_clusters.to_csv(f"{OUT_DIR}/ej2_clusters.csv", index=False)
    print(f"\nMejor k por silhouette: {mejor_k}")
    print(df_clusters.to_string(index=False))

    # -----------------------------------------------------------------
    # Heatmap de caracteristicas estandarizadas
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matriz_std.values, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(matriz_std.columns)))
    ax.set_xticklabels(matriz_std.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matriz_std.index)))
    ax.set_yticklabels(matriz_std.index)
    plt.colorbar(im, ax=ax, label="valor estandarizado (z)")
    ax.set_title("Heatmap de caracteristicas catch22 (estandarizadas)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ej2_heatmap.png", dpi=120)
    plt.close()

    # -----------------------------------------------------------------
    # Matriz de correlacion entre caracteristicas
    # -----------------------------------------------------------------
    corr = matriz.corr()
    corr.to_csv(f"{OUT_DIR}/ej2_corr_features.csv")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax, label="correlacion")
    ax.set_title("Correlacion entre caracteristicas catch22 (entre las 7 series)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ej2_corr.png", dpi=120)
    plt.close()

    # -----------------------------------------------------------------
    # Mapa de distancias entre series
    # -----------------------------------------------------------------
    dist = squareform(pdist(matriz_std.values, metric="euclidean"))
    df_dist = pd.DataFrame(dist, index=matriz.index, columns=matriz.index)
    df_dist.to_csv(f"{OUT_DIR}/ej2_distancias.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(df_dist.values, cmap="viridis")
    ax.set_xticks(range(len(df_dist.columns)))
    ax.set_xticklabels(df_dist.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df_dist.columns)))
    ax.set_yticklabels(df_dist.columns)
    for i in range(len(df_dist)):
        for j in range(len(df_dist)):
            ax.text(j, i, f"{df_dist.values[i, j]:.1f}", ha="center", va="center",
                     color="white", fontsize=7)
    plt.colorbar(im, ax=ax, label="distancia euclidiana")
    ax.set_title("Distancias entre series (espacio catch22 estandarizado)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ej2_distancias.png", dpi=120)
    plt.close()

    print("\nSerie mas atipica (mayor distancia media al resto):")
    dist_media = df_dist.mean().sort_values(ascending=False)
    print(dist_media.to_string())

    # -----------------------------------------------------------------
    # 2.12 Metricas de EDA para comparar contra catch22 
    # -----------------------------------------------------------------
    filas_eda = []
    for n in nombres:
        train = series_train[n]
        completa = completas[n]
        if (train <= 0).sum() > len(train) * 0.3:
            filas_eda.append({"serie": n, "label": labels[n], "fuerza_tendencia": np.nan,
                               "fuerza_estacionalidad": np.nan, "volatilidad_pct": np.nan,
                               "impacto_pandemia_pct": np.nan, "nota": "serie degenerada (>30% meses en cero)"})
            continue
        log_train = np.log1p(train)
        decomp = seasonal_decompose(log_train, model="additive", period=12)
        resid = decomp.resid.dropna()
        trend = decomp.trend.dropna()
        seasonal = decomp.seasonal.loc[resid.index]
        trend_r = trend.loc[resid.index]
        var_resid = resid.var()
        f_trend = max(0, 1 - var_resid / (trend_r + resid).var())
        f_seasonal = max(0, 1 - var_resid / (seasonal + resid).var())
        volatilidad = np.log1p(train).diff().dropna().std() * 100
        total_2019 = completa["2019-01":"2019-12"].sum()
        total_2020 = completa["2020-01":"2020-12"].sum()
        impacto_pandemia = (total_2020 / total_2019 - 1) * 100 if total_2019 else np.nan
        filas_eda.append({"serie": n, "label": labels[n], "fuerza_tendencia": f_trend,
                           "fuerza_estacionalidad": f_seasonal, "volatilidad_pct": volatilidad,
                           "impacto_pandemia_pct": impacto_pandemia, "nota": ""})

    tabla_eda = pd.DataFrame(filas_eda)
    tabla_eda.to_csv(f"{OUT_DIR}/ej2_eda_comparativo.csv", index=False)
    print("\nMetricas EDA (para comparar con catch22, punto 2.12):")
    print(tabla_eda.to_string(index=False))

    print("\nListo. Resultados guardados en resultados/ej2_*")


if __name__ == "__main__":
    main()
