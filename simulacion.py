import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# Parámetros de la simulación
lambdas = [2, 5, 10]  # tasas de llegada (paquetes por segundo)
n_observaciones = 1000

# Configurar gráficos
sns.set(style="whitegrid")

for lam in lambdas:

    datos = np.random.poisson(lam, n_observaciones)

    df = pd.DataFrame(datos, columns=["paquetes"])

    media = df["paquetes"].mean()
    varianza = df["paquetes"].var()

    print(f"λ = {lam} | Media: {media:.2f} | Varianza: {varianza:.2f}")

    conteo = df["paquetes"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=conteo.index, y=conteo.values, color="skyblue", label="Simulado")
    plt.plot(conteo.index, poisson.pmf(conteo.index, lam) * n_observaciones,
             color="red", marker="o", linestyle='dashed', label="Poisson teórica")
    plt.title(f"Distribución de Paquetes (λ = {lam})")
    plt.xlabel("Paquetes por segundo")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.show()
