from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chisquare, norm, poisson


def calcular_metricas_poisson(datos: Iterable[int]) -> dict:
    serie = np.asarray(list(datos))
    n = len(serie)
    media = float(np.mean(serie))
    varianza = float(np.var(serie, ddof=1))
    error_estandar = np.std(serie, ddof=1) / np.sqrt(n)
    z = norm.ppf(0.975)
    ic_media_95 = (float(media - z * error_estandar), float(media + z * error_estandar))

    return {
        "media": media,
        "varianza": varianza,
        "ic_media_95": ic_media_95,
    }


def prueba_chi_cuadrado_poisson(datos: Iterable[int], lam: float) -> dict:
    serie = np.asarray(list(datos))
    valores, frecuencias = np.unique(serie, return_counts=True)
    esperadas = poisson.pmf(valores, lam) * len(serie)

    mascara = esperadas > 0
    frecuencias = frecuencias[mascara]
    esperadas = esperadas[mascara]

    ajuste = frecuencias.sum() / esperadas.sum()
    esperadas = esperadas * ajuste

    chi2_stat, p_value = chisquare(f_obs=frecuencias, f_exp=esperadas)
    return {"chi2_stat": float(chi2_stat), "p_value": float(p_value)}


def ejecutar_monte_carlo(
    lambdas: Iterable[float], n_observaciones: int, n_corridas: int, rng: np.random.Generator
) -> pd.DataFrame:
    registros = []
    for lam in lambdas:
        medias = []
        varianzas = []
        for _ in range(n_corridas):
            muestra = rng.poisson(lam=lam, size=n_observaciones)
            medias.append(float(np.mean(muestra)))
            varianzas.append(float(np.var(muestra, ddof=1)))

        registros.append(
            {
                "lambda": lam,
                "media_promedio_corridas": np.mean(medias),
                "media_std_corridas": np.std(medias, ddof=1),
                "varianza_promedio_corridas": np.mean(varianzas),
                "varianza_std_corridas": np.std(varianzas, ddof=1),
            }
        )

    return pd.DataFrame(registros)
