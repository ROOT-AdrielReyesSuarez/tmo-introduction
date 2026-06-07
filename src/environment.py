import numpy as np
from typing import Tuple, Union

class DynamicEnvironment:
  """Clase encargada de gestionar el entorno dinámico de picos móviles cónicos.
    
    Se encarga de almacenar los parámetros del pico, evaluar soluciones en un
    espacio continuo y mutar el entorno aplicando un control estricto de límites.
    """

  def __init__(self, severity_c: float = 1.0, severity_w: float = 1.0, severity_h: float = 5.0) -> None:
    """Inicializa el entorno con un pico cónico en una posición aleatoria válida."""
    # Definición de los límites y fronteras del problema
    self.bounds_x: Tuple[float, float] = (-5.0, 5.0)
    self.bounds_w: Tuple[float, float] = (1.0, 12.0)
    self.bounds_h: Tuple[float, float] = (30.0, 70.0)

    # Guardamos las severidades de un ambiente a otro
    self.severity_c: float = severity_c
    self.severity_w: float = severity_w
    self.severity_h: float = severity_h

    # Inicialización aleatoria
    self.center_x: float = float(np.random.uniform(self.bounds_x[0], self.bounds_x[1]))
    self.center_y: float = float(np.random.uniform(self.bounds_x[0], self.bounds_x[1]))
    self.width: float = float(np.random.uniform(self.bounds_w[0], self.bounds_w[1]))
    self.height: float = float(np.random.uniform(self.bounds_h[0], self.bounds_h[1]))

  def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Calcula la aptitud (fitness) de las soluciones.
        
      Acepta tanto flotantes individuales como arrays de NumPy,
      lo que permite evaluar poblaciones enteras o mallas de coordenadas
      para gráficas de contorno a máxima velocidad sin usar bucles 'for'.
    """
    # Calculamos la distancai eucídea al cuadrado entre los puntos y el centro
    diff_x = x1 - self.center_x
    diff_y = x2 - self.center_y

    # Aplicamos la fórmula cónica matemática
    # f(x1, x2) = -w * sqrt((x1 - cx)**2 + (x2 - cy)**2) + h
    distance = np.sqrt(diff_x**2 + diff_y**2)
    fitness = -self.width * distance + self.height

    return fitness
  
  def get_optimum(self) -> Tuple[Tuple[float, float], float]:
    """Devuelve las coordenadas del óptimo real actual y su valor máximo de fitness."""
    return ((self.center_x, self.center_y), self.height)

  def mutate_enviroment(self) -> None:
    """Aplica ruido gaussiano a los parámetros del pico y controla sus fronteras.
        
    Este método simula la transición al siguiente ambiente independiente
    modificando la posición (center_x, center_y), el ancho (width) y la
    altura (height) del pico, recortando los valores si superan los límites.
    """
    # Mutamos la posición del centro utilizando las severidades individuales
    self.center_x += np.random.normal(0, self.severity_c)
    self.center_y += np.random.normal(0, self.severity_c)

    # Mutamos el ancho
    self.width += np.random.normal(0, self.severity_w)

    # Mutamos la altura
    self.height += np.random.normal(0, self.severity_h)

    # Controlamos que ningún parámetro viole los rangos establecidos
    self.center_x = float(np.clip(self.center_x, self.bounds_x[0], self.bounds_x[1]))
    self.center_y = float(np.clip(self.center_y, self.bounds_x[0], self.bounds_x[1]))
    self.width = float(np.clip(self.width, self.bounds_w[0], self.bounds_w[1]))
    self.height = float(np.clip(self.height, self.bounds_h[0], self.bounds_h[1]))
