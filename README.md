# Simulación de Tráfico de Red con Distribución de Poisson

Este proyecto simula la cantidad de paquetes que llegan a una red utilizando la **Distribución de Poisson**, y compara los resultados simulados con la distribución teórica. 

## Descripción

La simulación se realiza para diferentes tasas de llegada (`λ = 2, 5, 10`). Para cada caso:

- Se generan 1000 observaciones.
- Se calcula la media y la varianza de los datos simulados.
- Se compara la distribución simulada con la teórica usando un gráfico de barras.

## Tecnologías utilizadas

- Python 3
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scipy

## Cómo ejecutar

```bash
git clone https://github.com/Alejandrox51/simulacion-poisson-trafico-red

cd simulacion-poisson-trafico-red

pip install -r requisitos.txt

python simulacion.py
```

## Resultados esperados

Al ejecutar el script, se generan gráficos que comparan visualmente la distribución de Poisson teórica con los datos simulados, para cada valor de λ. Esto permite observar cómo se comporta el tráfico de red bajo distintos escenarios.

## Autor

Luis Alejandro Alcaraz Carrillo
Correo: luisalejandroa2208@gmail.com

