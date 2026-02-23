import os
import time
import numpy as np

from solvers.mcts_solver import MCTSSolver
from solvers.sa_solver import SASolver
from solvers.ga_solver import GASolver
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

    solver_params_mcts = {
        'iterations': 100_000_000_000, 
        'exploration_constant': 1.414, 
        'time_limit': 1800.0,
        'fitness_mode' : 1
    }

    solver_params_sa = {
        't_init': 10000.0,
        't_final': 0.001,
        'alpha': 0.99999,
        'time_limit': 600,
        'fitness_mode': 1
    }

    solver_params_ga_1 = {
        'pop_size': 2000,
        'generations': 5000,
        'tournament_size': 20,
        'mutation_rate': 0.3,
        'elitism_ratio': 0.05,
        'time_limit': 900.0,
        'fitness_mode' : 1
    }

    solver_params_ga = {
        'pop_size': 500,
        'generations': 1000,
        'tournament_size': 5,
        'mutation_rate': 0.25,
        'elitism_ratio': 0.05,
        'time_limit': None,
        'fitness_mode' : 1
    }


    
    start_time = time.time()
    best_path, best_score = ParallelRunner.run(MCTSSolver, solver_params_mcts, cylinders)
    elapsed = time.time() - start_time
    
    print(f"\n=== RÉSULTATS DE L'OPTIMISATION ===")
    print(f"Temps de calcul total : {elapsed:.2f} secondes")
    print(f"Meilleur score absolu : {best_score:_.2f}")
    print(f"Ordre de visite : {best_path}")
        
    translator = RobotTranslator(cylinders, start_x=0.0, start_y=0.0, start_angle=0.0)
    
    os.makedirs("results", exist_ok=True)
    translator.generate_script(best_path, f"results/script_robot_mcts.txt")

    RouteVisualizer.plot_trajectory(cylinders, best_path, save_path="results/map_solution.png")

if __name__ == "__main__":
    main()