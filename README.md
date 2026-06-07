# Simulador de Optimización en Entornos Dinámicos (PSO vs. DE)

Este repositorio contiene un simulador desarrollado en Python para comparar el rendimiento de dos populares metaheurísticas de optimización continua: la **Optimización por Enjambre de Partículas (PSO)** y la **Evolución Diferencial (DE)**, en un entorno dinámico continuo que evoluciona en el tiempo.

El problema consiste en buscar y rastrear el óptimo global de una función que representa un **pico móvil cónico tridimensional** que sufre variaciones aleatorias (gaussianas) de ubicación, anchura y altura en cada cambio ambiental.

---

## 🎯 Objetivo del Proyecto

El propósito de este código es evaluar la capacidad de adaptación de optimizadores continuos frente a cambios dinámicos del terreno (mutaciones ambientales). El simulador ejecuta:
1. Una secuencia temporal de **ambientes independientes** modificados por perturbaciones gaussianas (simulando eventos dinámicos).
2. La optimización en cada paso ambiental para localizar la coordenada global del pico más alto.
3. El cálculo de los errores eucídeos medios frente al óptimo real.
4. La generación de gráficos de contorno de alta definición que visualizan el comportamiento de búsqueda de cada algoritmo.

---

## 📂 Estructura del Repositorio

El código se organiza en los siguientes módulos dentro del directorio `src/`:

* **[environment.py](src/environment.py)**: Define la clase `DynamicEnvironment`. Gestiona el pico cónico, su evaluación matemática tridimensional vectorizada y el método `mutate_enviroment` que simula la transición a nuevos estados de ambiente aplicando ruido gaussiano sobre la ubicación del centro, el ancho y la altura del pico.
* **[optimizers.py](src/optimizers.py)**: Contiene la interfaz abstracta `OptimizerStrategy` y las implementaciones de los optimizadores:
  * **PSOStrategy (Particle Swarm Optimization)**: Optimiza ajustando velocidades y posiciones de un enjambre de partículas en función de sus mejores marcas históricas individuales (componente cognitivo) y la mejor marca del enjambre (componente social).
  * **DifferentialEvolutionStrategy (Differential Evolution)**: Optimiza utilizando un enfoque evolutivo basado en la mutación diferencial de vectores (`DE/rand/1`), cruce binomial y selección codiciosa.
* **[main.py](src/main.py)**: Script principal para ejecutar el simulador competitivo. Corre la simulación a lo largo de 5 ambientes, evalúa el error medio de cada optimizador y genera una figura visual comparativa en `comparativa_pso_vs_de.png`.
* **[requirements.txt](requirements.txt)**: Archivo con las dependencias del entorno de ejecución.

---

## 🛠️ Requisitos e Instalación

Para configurar y ejecutar el proyecto localmente, sigue los siguientes pasos:

### 1. Clonar el repositorio
Abre una terminal y clona este repositorio en tu máquina local:
```bash
git clone <URL_DEL_REPOSITORIO>
cd tmo-introduction
```
> [!NOTE]
> *Reemplaza `<URL_DEL_REPOSITORIO>` por la URL de clonado correspondiente (HTTPS/SSH).*

### 2. Generar el entorno virtual
Se recomienda utilizar un entorno virtual de Python para aislar las librerías del proyecto.

En **Linux/macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

En **Windows** (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias y configurar las versiones
Una vez activado el entorno virtual, instala las librerías necesarias ejecutando:
```bash
pip install -r requirements.txt
```

---

## 📦 Librerías y Versiones Utilizadas

Las dependencias requeridas están especificadas con versiones estrictas para garantizar la compatibilidad y evitar comportamientos inesperados:

| Librería | Versión | Descripción |
| :--- | :--- | :--- |
| **`numpy`** | `2.4.6` | Operaciones matriciales y cálculos matemáticos vectorizados de alta eficiencia. |
| **`matplotlib`** | `3.10.9` | Generación del lienzo gráfico comparativo y mapas de contornos. |
| **`pillow`** | `12.2.0` | Soporte para el manejo y guardado de archivos de imagen. |
| **`contourpy`** | `1.3.3` | Dependencia interna para el trazado rápido de curvas de contornos en 2D. |
| **`cycler`** | `0.12.1` | Manejo de secuencias de estilos cíclicos en gráficos. |
| **`fonttools`** | `4.63.0` | Utilidad de tipografías para el renderizado del lienzo de Matplotlib. |
| **`kiwisolver`** | `1.5.0` | Solucionador geométrico utilizado internamente por Matplotlib para el layout de los ejes. |
| **`packaging`** | `26.2` | Utilidades de distribución de paquetes Python. |
| **`pyparsing`** | `3.3.2` | Análisis sintáctico de expresiones de texto. |
| **`python-dateutil`** | `2.9.0.post0` | Manejo avanzado de fechas y marcas temporales. |
| **`six`** | `1.17.0` | Librería de compatibilidad de código Python 2 y 3. |

---

## 🚀 Cómo Ejecutar el Simulador

Desde la raíz del repositorio (`tmo-introduction`), con tu entorno virtual activo, ejecuta el siguiente comando:

```bash
python -m src.main
```

### Salida por Consola
El programa imprimirá la evolución de la simulación paso a paso:

```text
=== Inicializando Simulador Competitivo TMO ===

Iniciando carrera temporal en 5 ambientes consecutivos...
---------------------------------------------------------------------------
[Ambiente 1]
   -> Err PSO: 0.0345 | Err DE: 0.0121
[Ambiente 2]
   -> Err PSO: 0.1432 | Err DE: 0.0543
[Ambiente 3]
   -> Err PSO: 0.0987 | Err DE: 0.0211
[Ambiente 4]
   -> Err PSO: 0.2312 | Err DE: 0.0892
[Ambiente 5]
   -> Err PSO: 0.1764 | Err DE: 0.0430
---------------------------------------------------------------------------
=== Cómputo final del experimento ===
Error medio temporal del PSO: 0.1368
Error medio temporal de la DE: 0.0439

¡Imagen guardada con éxito como 'comparativa_pso_vs_de.png'!
```

### Gráfico Comparativo Guardado
Al finalizar, se genera una imagen `comparativa_pso_vs_de.png` que consta de:
* **Fila Superior**: Visualización del comportamiento del enjambre PSO.
* **Fila Inferior**: Visualización del comportamiento de Evolución Diferencial.
* **Columnas**: Cada uno de los 5 ambientes simulados con sus curvas de nivel en colores y:
  * Una **estrella roja (`*`)** que indica el óptimo real global del ambiente.
  * Una **cruz cian/dorada (`P`)** que indica el mejor punto encontrado por PSO y DE respectivamente en dicho ambiente.
