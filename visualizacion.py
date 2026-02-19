from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

sns.set(style="whitegrid")


def graficar_distribucion_poisson(datos, lam: float, output_path: Path, show: bool = True) -> None:
    conteo_x = sorted(set(datos))
    conteo_y = [sum(datos == k) for k in conteo_x]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=conteo_x, y=conteo_y, color="skyblue", label="Simulado")
    plt.plot(
        conteo_x,
        poisson.pmf(conteo_x, lam) * len(datos),
        color="red",
        marker="o",
        linestyle="dashed",
        label="Poisson teórica",
    )
    plt.title(f"Distribución de Paquetes (λ = {lam})")
    plt.xlabel("Paquetes por segundo")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
