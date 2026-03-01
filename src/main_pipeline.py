import numpy as np

from solvers.sa_solver import SASolver
from solvers.ga_solver import GASolver
from solvers.mcts_solver import MCTSSolver
from solvers.beam_solver import BeamSearchSolver
from solvers.memetic_solver import MemeticSolver
from solvers.weight_ratio_solver import WeightedRatioSolver
from pipeline import EvaluationPipeline

def load_real_instance(filepath):
    """Charge la map"""
    raw_data = np.loadtxt(filepath)
    cylinders = np.zeros((len(raw_data), 4), dtype=np.float64)
    cylinders[:, 0] = raw_data[:, 0] # x
    cylinders[:, 1] = raw_data[:, 1] # y
    cylinders[:, 2] = raw_data[:, 2] # masse
    cylinders[:, 3] = 2 * raw_data[:, 2] - 1 # points (2x - 1)
    return cylinders

def main():
    DATA_DIR = "data"  
    RESULTS_DIR = "results"    
    UNITY_EXE = "C:/Users/Utilisateur/Documents/CoursCI2/Challenge/RunTime-2026-ultimate/challenge-robotique.exe"

    #target_map = "donnees-map2.txt"
    target_map = None
    
    pipeline = EvaluationPipeline(DATA_DIR, RESULTS_DIR, UNITY_EXE)

    
    pipeline.add_solver(
        name="SA",
        solver_class=SASolver,
        params={
            't_init': 10000.0, 't_final': 0.001, 'alpha': 0.99999, 
            'time_limit': 900.0, 'fitness_mode': 0
        }
    )
    
    pipeline.add_solver(
        name="BeamSearch",
        solver_class=BeamSearchSolver,
        params={
            "beam_width" : 500_000,
            "fitness_mode" : 0
        }
    )

    pipeline.add_solver(
        name="GA",
        solver_class=GASolver,
        params={
            "pop_size" : 2000, "generations" : 5000, "tournament_size" : 20, 
            "mutation_rate" : 0.3, "elitism_ratio" : 0.05, "time_limit" : 900,
            "fitness_mode" : 0
        }
    )

    pipeline.add_solver(
        name="MCTS",
        solver_class=MCTSSolver,
        params={"iterations" : 1_000_000_000, "exploration_constant" : 1.414,
            "time_limit" : 1800, "fitness_mode" : 0
        }
    )
    
    pipeline.add_solver(
        name="Memetic",
        solver_class=MemeticSolver,
        params={
            'pop_size': 600,           
            'generations': 600,        
            'tournament_size': 6,      
            'mutation_rate': 0.2,      
            'ls_rate': 0.8,            
            'ls_max_steps': 50,        
            'elitism_ratio': 0.1,      
            'time_limit': 900.0,       
            'fitness_mode': 0          
        }
    )


    pipeline.add_solver(
        name="Ratio_Original_49",
        solver_class=WeightedRatioSolver,
        params={'wp': 1.0, 'wd': 1.0, 'wm': 0.0}
    )

    pipeline.add_solver(
        name="Ratio_Mass_Aversion",
        solver_class=WeightedRatioSolver,
        params={'wp': 1.0, 'wd': 1.0, 'wm': 1.5}
    )

    pipeline.add_solver(
        name="Ratio_Distance_Hater",
        solver_class=WeightedRatioSolver,
        params={'wp': 1.0, 'wd': 2.0, 'wm': 0.5}
    )

    pipeline.add_solver(
        name="Ratio_Point_Lover",
        solver_class=WeightedRatioSolver,
        params={'wp': 2.0, 'wd': 1.0, 'wm': 0.5}
    )

    # Lancement de toute la batterie de tests
    if target_map:
        cylinders = load_real_instance("data/" + target_map)
        print(f"Maximum théorique : {np.sum(cylinders[:, 3])}")
    pipeline.run_all(load_real_instance, target_map)
    
    print("\nPIPELINE TERMINÉ. Vérifie le fichier benchmark_results.csv")

if __name__ == "__main__":
    main()