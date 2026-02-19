import argparse
from pathlib import Path

import numpy as np

from analisis import (
    calcular_metricas_poisson,
    ejecutar_monte_carlo,
    prueba_chi_cuadrado_poisson,
)
from cola import simular_mm1
from visualizacion import graficar_distribucion_poisson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulación de tráfico de red con Poisson + análisis Monte Carlo + modelo M/M/1"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[2, 5, 10],
        help="Lista de tasas de llegada λ para Poisson.",
    )
    parser.add_argument(
        "--observaciones",
        type=int,
        default=1000,
        help="Número de observaciones por simulación.",
    )
    parser.add_argument(
        "--corridas",
        type=int,
        default=100,
        help="Número de corridas Monte Carlo por λ.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados"),
        help="Carpeta para guardar CSV y gráficas.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar gráficos en pantalla (solo guardar).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=12.0,
        help="Tasa de servicio μ para simulación M/M/1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    resumen_montecarlo = ejecutar_monte_carlo(
        lambdas=args.lambdas,
        n_observaciones=args.observaciones,
        n_corridas=args.corridas,
        rng=rng,
    )
    resumen_montecarlo.to_csv(args.output_dir / "resumen_montecarlo.csv", index=False)

    detalles = []
    for lam in args.lambdas:
        datos = rng.poisson(lam=lam, size=args.observaciones)
        metricas = calcular_metricas_poisson(datos)
        chi2 = prueba_chi_cuadrado_poisson(datos, lam)

        print(
            (
                f"λ={lam:.2f} | media={metricas['media']:.3f} "
                f"(IC95% {metricas['ic_media_95'][0]:.3f}, {metricas['ic_media_95'][1]:.3f}) | "
                f"varianza={metricas['varianza']:.3f} | p-valor chi²={chi2['p_value']:.4f}"
            )
        )

        grafica_path = args.output_dir / f"poisson_lambda_{str(lam).replace('.', '_')}.png"
        graficar_distribucion_poisson(
            datos=datos,
            lam=lam,
            output_path=grafica_path,
            show=not args.no_show,
        )

        detalles.append(
            {
                "lambda": lam,
                "media": metricas["media"],
                "varianza": metricas["varianza"],
                "ic_media_95_min": metricas["ic_media_95"][0],
                "ic_media_95_max": metricas["ic_media_95"][1],
                "chi2_stat": chi2["chi2_stat"],
                "chi2_p_value": chi2["p_value"],
            }
        )

    mm1 = simular_mm1(
        lambda_llegadas=args.lambdas[-1],
        mu_servicio=args.mu,
        n_clientes=args.observaciones,
        rng=rng,
    )
    print(
        (
            "M/M/1 "
            f"(λ={args.lambdas[-1]:.2f}, μ={args.mu:.2f}) | "
            f"utilización={mm1['utilizacion']:.3f} | "
            f"espera_promedio={mm1['espera_promedio']:.3f} s | "
            f"tiempo_sistema_promedio={mm1['tiempo_sistema_promedio']:.3f} s"
        )
    )

    import pandas as pd

    pd.DataFrame(detalles).to_csv(args.output_dir / "resumen_detallado.csv", index=False)
    pd.DataFrame([mm1]).to_csv(args.output_dir / "mm1_resumen.csv", index=False)


if __name__ == "__main__":
    main()
