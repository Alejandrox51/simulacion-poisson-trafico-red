from __future__ import annotations

import numpy as np


def simular_mm1(
    lambda_llegadas: float,
    mu_servicio: float,
    n_clientes: int,
    rng: np.random.Generator,
) -> dict:
    if lambda_llegadas <= 0 or mu_servicio <= 0:
        raise ValueError("λ y μ deben ser mayores a cero")

    interarrivals = rng.exponential(scale=1 / lambda_llegadas, size=n_clientes)
    servicios = rng.exponential(scale=1 / mu_servicio, size=n_clientes)

    llegadas = np.cumsum(interarrivals)
    inicios = np.zeros(n_clientes)
    salidas = np.zeros(n_clientes)

    for i in range(n_clientes):
        if i == 0:
            inicios[i] = llegadas[i]
        else:
            inicios[i] = max(llegadas[i], salidas[i - 1])
        salidas[i] = inicios[i] + servicios[i]

    esperas = inicios - llegadas
    tiempos_sistema = salidas - llegadas
    utilizacion = float(np.sum(servicios) / salidas[-1])

    cola_en_llegada = []
    for i in range(n_clientes):
        en_sistema = np.sum((llegadas[: i + 1] <= llegadas[i]) & (salidas[: i + 1] > llegadas[i]))
        cola_en_llegada.append(max(int(en_sistema - 1), 0))

    return {
        "lambda": lambda_llegadas,
        "mu": mu_servicio,
        "rho_teorico": lambda_llegadas / mu_servicio,
        "utilizacion": utilizacion,
        "espera_promedio": float(np.mean(esperas)),
        "tiempo_sistema_promedio": float(np.mean(tiempos_sistema)),
        "cola_maxima": int(np.max(cola_en_llegada)),
    }
