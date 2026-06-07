from abc import ABC, abstractmethod
import numpy as np
from src.environment import DynamicEnviroment

class OptimizerStrategy(ABC):
  """Interfaz abstracta que define el contrato obligatorio para cualquier optimizador.
  
  Todas las metaheurísticas (PSO, DE, etc.) deberán heredar de esta clase
  e implementar obligatoriamente sus métodos.
  """
  
  @abstractmethod
  def minimize_or_maximize(self, env: DynamicEnviroment, generations: int) -> np.ndarray:
      """Ejecuta el proceso de optimización sobre el entorno en su estado estático actual.
      
      Args:
        env: Instancia del entorno dinámico (el pico móvil cónico).
        generations: Número de iteraciones internas que tiene el algoritmo para buscar.
          
      Returns:
        np.ndarray: Un array de NumPy con las coordenadas [x1, x2] de la mejor 
                    solución encontrada por el algoritmo.
      """
      pass

class PSOStrategy(OptimizerStrategy):
  """Estrategia concreta que implementa el algoritmo de Optimización por 

  Enjambre de Partículas (PSO) para resolver el entorno dinámico.
  """
  
  # Valores tomados de estudios de Clerc y Kennedy
  def __init__(self, num_particles: int = 30, w_inertia: float = 0.729, c1_cognitive: float = 1.49443, c2_social: float = 1.49443) -> None:
    """Inicializa los hiperparámetros de control del enjambre."""
    self.num_particles: int = num_particles
    
    # Hiperparámetros de la mente de las partículas
    self.w_inertia: float = w_inertia
    self.c1_cognitive: float = c1_cognitive
    self.c2_social: float = c2_social
  
  def minimize_or_maximize(self, env: DynamicEnviroment, generations: int) -> np.ndarray:
    """Ejecuta el algoritmo PSO para encontrar el pico más alto del entorno."""
    # 1. Extraemos los límites espaciales del entorno para saber dónde movernos
    min_x, max_x = env.bounds_x
    
    # 2. Inicialización de Posiciones usando la Distribución Uniforme
    # Cada fila es una partícula; la columna 0 es su coordenada X1 y la columna 1 es su X2
    positions = np.random.uniform(min_x, max_x, (self.num_particles, 2))
    
    # 3. Inicialización de Velocidades en cero
    # Al arrancar, las partículas están quietas esperando a evaluar el terreno
    velocities = np.zeros((self.num_particles, 2))
    
    # 4. Evaluación inicial del fitness (altitud) de cada partícula
    # Pasamos las columnas completas de forma vectorizada a nuestro entorno
    p_best_fitness = env.evaluate(positions[:, 0], positions[:, 1])
    
    # 5. Memoria Individual: Al principio, la mejor posición histórica de cada partícula es su posición actual
    p_best_positions = np.copy(positions)
    
    # 6. Memoria Social: Buscamos cuál de todas las partículas ha caído en el punto más alto
    best_idx = np.argmax(p_best_fitness)
    g_best_position = np.copy(p_best_positions[best_idx])
    g_best_fitness = p_best_fitness[best_idx]

    # 7. Bucle principal de optimización
    for gen in range(generations):
      
      # Generamos coeficientes aleatorios independientes para el componente cognitivo y social
      # Creamos matrices de (num_particles, 2) con números al azar entre 0 y 1
      r1 = np.random.uniform(0.0, 1.0, (self.num_particles, 2))
      r2 = np.random.uniform(0.0, 1.0, (self.num_particles, 2))
      
      # Actualización de velocidades de todo el enjambre de golpe (Vectorizado)
      velocities = (self.w_inertia * velocities + 
                    self.c1_cognitive * r1 * (p_best_positions - positions) + 
                    self.c2_social * r2 * (g_best_position - positions))
      
      # Actualización de posiciones (Dar el paso físico en el mapa)
      positions = positions + velocities
      
      # Control estricto de fronteras para las partículas (Clamping espacial)
      # Evitamos que los exploradores se salgan del mapa de búsqueda de tu enunciado [-5.0, 5.0]
      positions = np.clip(positions, min_x, max_x)
      
      # 7. Evaluación del nuevo terreno tras el movimiento
      current_fitness = env.evaluate(positions[:, 0], positions[:, 1])
      
      # FÓRMULA 3: Actualización de la Memoria Individual (PBest)
      # Comparamos elemento a elemento si la nueva altura es mejor que su récord histórico
      improved_mask = current_fitness > p_best_fitness
      p_best_positions[improved_mask] = positions[improved_mask]
      p_best_fitness[improved_mask] = current_fitness
      
      # FÓRMULA 4: Actualización de la Memoria Social (GBest)
      # Buscamos si en este turno alguien ha superado el récord absoluto del enjambre
      current_best_idx = np.argmax(current_fitness)
      if current_fitness[current_best_idx] > g_best_fitness:
        g_best_fitness = current_fitness[current_best_idx]
        g_best_position = np.copy(positions[current_best_idx])
        
    # Al terminar todas las generaciones, devolvemos la mejor posición histórica del grupo
    return g_best_position