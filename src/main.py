import numpy as np
import matplotlib.pyplot as plt
from src import DynamicEnvironment
# Importamos ambas estrategias para la comparativa
from src.optimizers import PSOStrategy, DifferentialEvolutionStrategy

if __name__ == "__main__":
  print("=== Inicializando Simulador Competitivo TMO ===")
  
  # Instanciamos el entorno único
  env = DynamicEnvironment()
  
  # Instanciamos ambos optimizadores con el mismo tamaño de población (30)
  pso_opt = PSOStrategy(num_particles=30)
  de_opt = DifferentialEvolutionStrategy(pop_size=30)
  
  # Configuración de la simulación temporal
  total_environments: int = 5 
  generations_per_env: int = 50
  
  pso_errors: list[float] = []
  de_errors: list[float] = []

  # Rejilla continua de alta definición para dibujar las curvas de nivel
  x = np.linspace(-5.0, 5.0, 100)
  y = np.linspace(-5.0, 5.0, 100)
  X, Y = np.meshgrid(x, y)
  
  # Creamos el lienzo: 2 filas (Fila 0 = PSO, Fila 1 = DE) y 5 columnas (Ambientes)
  fig, axes = plt.subplots(2, total_environments, figsize=(20, 8))

  print(f"\nIniciando carrera temporal en {total_environments} ambientes consecutivos...")
  print("-" * 75)

  # Bucle principal de la línea de tiempo
  for env_idx in range(total_environments):
    if env_idx > 0:
      env.mutate_enviroment()  # Provocamos el terremoto en la isla
      
    print(f"[Ambiente {env_idx + 1}]")
    
    # Obtenemos el punto más alto real en este instante
    (real_x, real_y), max_fit = env.get_optimum()
    Z = env.evaluate(X, Y)
    
    # -----------------------------------------------------------------------
    # Algoritmo PSO
    # -----------------------------------------------------------------------
    ax_pso = axes[0, env_idx]
    ax_pso.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax_pso.scatter(real_x, real_y, color='red', marker='*', s=120, label='Real' if env_idx == 0 else "")
    
    # El PSO ejecuta su búsqueda estática congelada
    sol_pso = pso_opt.minimize_or_maximize(env, generations=generations_per_env)
    ax_pso.scatter(sol_pso[0], sol_pso[1], color='cyan', marker='P', s=90, label='PSO' if env_idx == 0 else "")
    
    err_pso = np.sqrt((sol_pso[0] - real_x)**2 + (sol_pso[1] - real_y)**2)
    pso_errors.append(err_pso)
    
    # Estética fila PSO
    ax_pso.set_title(f"Ambiente {env_idx + 1} - PSO\nErr: {err_pso:.3f}", fontsize=10)
    ax_pso.set_xticks([])
    if env_idx % total_environments != 0: ax_pso.set_yticks([])
    if env_idx == 0: ax_pso.legend(loc='upper right', fontsize=8)
    ax_pso.grid(True, alpha=0.15)

    # -----------------------------------------------------------------------
    # Algoritmo Evolución Diferencial (DE)
    # -----------------------------------------------------------------------
    ax_de = axes[1, env_idx]
    ax_de.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax_de.scatter(real_x, real_y, color='red', marker='*', s=120)
    
    # La Evolución Diferencial ejecuta su búsqueda en el mismo terreno
    sol_de = de_opt.minimize_or_maximize(env, generations=generations_per_env)
    ax_de.scatter(sol_de[0], sol_de[1], color='gold', marker='P', s=90, label='DE' if env_idx == 0 else "")
    
    err_de = np.sqrt((sol_de[0] - real_x)**2 + (sol_de[1] - real_y)**2)
    de_errors.append(err_de)
    
    # Estética fila DE
    ax_de.set_title(f"Ambiente {env_idx + 1} - DE\nErr: {err_de:.3f}", fontsize=10)
    if env_idx % total_environments != 0: ax_de.set_yticks([])
    if env_idx == 0: ax_de.legend(loc='upper right', fontsize=8)
    ax_de.grid(True, alpha=0.15)

    print(f"   -> Err PSO: {err_pso:.4f} | Err DE: {err_de:.4f}")

  # Imprimir balance de resultados por consola
  print("-" * 75)
  print("=== Cómputo final del experimento ===")
  print(f"Error medio temporal del PSO: {np.mean(pso_errors):.4f}")
  print(f"Error medio temporal de la DE: {np.mean(de_errors):.4f}")
  
  # Guardamos el lienzo comparativo
  plt.tight_layout()
  plt.savefig("comparativa_pso_vs_de.png", dpi=300)
  print("\n¡Imagen guardada con éxito como 'comparativa_pso_vs_de.png'!")