import time
from .base_solver import BaseSolver
from utils_solver import simulated_annealing_core

class SASolver(BaseSolver):
    def __init__(self, t_init=10000.0, t_final=0.001, alpha=0.99999, time_limit=900.0, fitness_mode=0):

        self.t_init = t_init
        self.t_final = t_final
        self.alpha = alpha
        self.time_limit = time_limit
        self.fitness_mode = fitness_mode

    def solve(self, cylinders):
        global_best_score = -float('inf')
        global_best_path = None
        
        start_time = time.time()
        restarts = 0
        
        while time.time() - start_time < self.time_limit:
            score, path = simulated_annealing_core(
                cylinders,
                fitness_mode=self.fitness_mode, 
                T_init=self.t_init, 
                T_final=self.t_final, 
                alpha=self.alpha
            )
            
            if score > global_best_score:
                global_best_score = score
                global_best_path = path.copy()
                
            restarts += 1
            
        print(f"{restarts} cycles de Recuit Simulé effectués sur ce coeur")
        return global_best_path.tolist(), global_best_score