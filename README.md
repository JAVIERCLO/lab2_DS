# lab2_DS
Laboratorio 2 de Data Science sobre LSTM y catch22

## Informe

El informe final, con las respuestas a todos los puntos de las instrucciones, está en [`Informe.ipynb`](Informe.ipynb).

## Estructura

- `datos_comunes.py` — construcción de las series y partición train/test (idéntica a la del Laboratorio 1).
- `Modelos.ipynb` — Ejercicio 1: modelos LSTM con tuneo de parámetros para dos series.
- `Catch22.ipynb` — Ejercicio 2: extracción de características catch22, PCA, clustering, y LSTM con características catch22.
- `Informe.ipynb` — informe final, responde en orden cada punto de las instrucciones a partir de los resultados guardados en `resultados/`.
- `resultados/` — métricas, matrices y gráficas generadas por los dos notebooks anteriores (usadas por el informe).
- `datos/` — datos crudos (Excel del Ministerio, mismo archivo que el Laboratorio 1).

## Reproducir

Para regenerar todo desde cero: correr `Modelos.ipynb` y luego `Catch22.ipynb` de principio a fin (ambos guardan sus resultados en `resultados/`), y después `Informe.ipynb`.

Requiere `pycatch22`, que en Windows necesita compilar una extensión en C — con Visual Studio Build Tools o con MinGW (`pip install pycatch22`, especificando el compilador si hace falta).
