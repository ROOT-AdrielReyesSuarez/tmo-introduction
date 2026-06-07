import numpy as np
import matplotlib.pyplot as plt
from src import DynamicEnvironment

if __name__ == '__main__':
  print("=== Inicializando Simulador TMO - Paso 1 ===")
  # Instanciamos el entorno dinámico con sus valores por defecto
  env = DynamicEnvironment()

  # Creamos el espacio continuo discretizado para la representación visual
  # Generamos 200 puntos equidistantes entre -5.0 y 5.0 para cada eje
  x = np.linspace(-5.0, 5.0, 200)
  y = np.linspace(-5.0, 5.0, 200)

  # Creamos una malla de coordenadas
  X, Y = np.meshgrid(x, y)

  # Evaluamos la malla
  Z = env.evaluate(X, Y)

  # Creamos una ventana gráfica con dos sub-gráficas
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

  # --- AMBIENTE 1 (Izquierda) ---
  # Pintamos las curvas de nivel rellenas de color (Contour Plot)
  contour1 = ax1.contourf(X, Y, Z, levels=30, cmap='viridis')
  fig.colorbar(contour1, ax=ax1, label='Fitness (Altitud)')
  
  # Obtenemos el óptimo real y lo dibujamos como una estrella roja
  (opt_x, opt_y), opt_fit = env.get_optimum()
  ax1.scatter(opt_x, opt_y, color='red', marker='*', s=200, label=f'Óptimo Real\n({opt_x:.2f}, {opt_y:.2f})')
  
  ax1.set_title(f"Ambiente Inicial (Ambiente 1)\nAncho: {env.width:.2f} | Altura: {env.height:.2f}")
  ax1.set_xlabel("Variable X1")
  ax1.set_ylabel("Variable X2")
  ax1.legend(loc='upper right')
  ax1.grid(True, alpha=0.3)
  
  # Provocamos la transición temporal (Mutación del entorno)
  env.mutate_enviroment()
  
  # Re-evaluamos la misma malla de puntos con las nuevas propiedades de la montaña
  Z_mutado = env.evaluate(X, Y)
  
  # --- AMBIENTE 2 (Derecha) ---
  contour2 = ax2.contourf(X, Y, Z_mutado, levels=30, cmap='viridis')
  fig.colorbar(contour2, ax=ax2, label='Fitness (Altitud)')
  
  # Obtenemos las nuevas coordenadas del óptimo mutado
  (new_opt_x, new_opt_y), new_opt_fit = env.get_optimum()
  ax2.scatter(new_opt_x, new_opt_y, color='red', marker='*', s=200, label=f'Nuevo Óptimo\n({new_opt_x:.2f}, {new_opt_y:.2f})')
  
  ax2.set_title(f"Siguiente Ambiente (Ambiente 2)\nAncho: {env.width:.2f} | Altura: {env.height:.2f}")
  ax2.set_xlabel("Variable X1")
  ax2.set_ylabel("Variable X2")
  ax2.legend(loc='upper right')
  ax2.grid(True, alpha=0.3)
  
  # Guardamos el resultado como una imagen en la raíz del proyecto
  plt.tight_layout()
  plt.savefig("resultado_paso1.png", dpi=300)
  print("¡Éxito! Los mapas topográficos se han guardado en 'resultado_paso1.png'")

  