# Simulación de Tráfico de Red con Distribución de Poisson

Este proyecto modela la llegada de paquetes de red con una distribución de **Poisson**, evalúa el ajuste estadístico de las muestras simuladas y extiende el análisis con una cola **M/M/1**.

## ¿Qué incluye esta versión?

1. **Simulación Poisson** para múltiples tasas `λ`.
2. **Monte Carlo** con múltiples corridas por cada `λ`.
3. **Métricas estadísticas** por muestra:
   - media
   - varianza
   - intervalo de confianza al 95% de la media
4. **Prueba de bondad de ajuste Chi-cuadrado** para comparar simulación vs. teoría.
5. **Simulación de cola M/M/1** con métricas de desempeño.
6. **Salida a archivos**:
   - gráficas en PNG
   - resúmenes en CSV

## Tecnologías utilizadas

- Python 3
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scipy

## Instalación

```bash
pip install -r requisitos.txt
```

## Uso básico

```bash
python simulacion.py
```

## Uso avanzado

```bash
python simulacion.py \
  --lambdas 2 5 10 \
  --observaciones 1000 \
  --corridas 100 \
  --seed 42 \
  --mu 12 \
  --output-dir resultados \
  --no-show
```

### Parámetros disponibles

- `--lambdas`: lista de tasas de llegada.
- `--observaciones`: tamaño de muestra por simulación.
- `--corridas`: número de corridas Monte Carlo por `λ`.
- `--seed`: semilla aleatoria para reproducibilidad.
- `--mu`: tasa de servicio para la simulación M/M/1.
- `--output-dir`: carpeta de salida para archivos.
- `--no-show`: evita abrir ventanas de gráficos (útil en servidores/CI).

## Archivos de salida

En `output-dir` se generan:

- `resumen_montecarlo.csv`
- `resumen_detallado.csv`
- `mm1_resumen.csv`
- `poisson_lambda_<valor>.png`

## Estructura del proyecto

- `simulacion.py`: CLI principal y orquestación.
- `analisis.py`: métricas, IC y prueba Chi-cuadrado.
- `visualizacion.py`: generación de gráficas.
- `cola.py`: simulador de cola M/M/1.

## Autor

Luis Alejandro Alcaraz Carrillo
Correo: luisalejandroa2208@gmail.com
