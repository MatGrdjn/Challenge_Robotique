import os
import time
import numpy as np

from solvers.mcts_solver import MCTSSolver
from solvers.parallel_runner import ParallelRunner

from visualizer import RouteVisualizer
from robot_translator import RobotTranslator


def load_real_instance(filepath):
    raw_data = np.loadtxt(filepath)
    cylinders = np.zeros((len(raw_data), 4), dtype=np.float64)
    cylinders[:, 0] = raw_data[:, 0] # x
    cylinders[:, 1] = raw_data[:, 1] # y
    cylinders[:, 2] = raw_data[:, 2] # masse
    cylinders[:, 3] = 2 * raw_data[:, 2] - 1 # points
    return cylinders



def main():
    print("--- DÉMARRAGE DU SOLVER ---")
    
    cylinders = load_real_instance("data/donnees-map1.txt")

    solver_params = {
        'iterations': 1_000_000, 
        'exploration_constant': 1.414, 
        'time_limit': 180.0
    }

    
    start_time = time.time()
    best_path, best_score = ParallelRunner.run(MCTSSolver, solver_params, cylinders)
    elapsed = time.time() - start_time
    
    print(f"\n=== RÉSULTATS DE L'OPTIMISATION ===")
    print(f"Temps de calcul total : {elapsed:.2f} secondes")
    print(f"Meilleur score absolu : {best_score:_.2f}")
    print(f"Ordre de visite : {best_path}")
        

        
    translator = RobotTranslator(start_x=0.0, start_y=0.0, start_angle=0.0)
    translator.load_path([(cylinders[i, 0], cylinders[i, 1]) for i in best_path])
    
    os.makedirs("results", exist_ok=True)
    translator.export_to_txt("results/script_robot.txt")

    RouteVisualizer.plot_trajectory(cylinders, best_path, save_path="results/map_solution.png")


if __name__ == "__main__":
    main()