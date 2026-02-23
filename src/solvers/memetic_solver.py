import time
from .base_solver import BaseSolver
from utils_solver import memetic_algorithm_core

class MemeticSolver(BaseSolver):
    def __init__(self, pop_size=100, generations=200, tournament_size=5, mutation_rate=0.2, ls_rate=1.0, ls_max_steps=30, elitism_ratio=0.1, time_limit=900.0, fitness_mode=1):
        """
        :param mutation_rate: Probabilité de subir une mutation aléatoire avant la recherche locale.
        :param ls_rate: Probabilité qu'un enfant fasse une Recherche Locale (1.0 = tous)
        :param ls_max_steps: Nombre max d'améliorations par descente de gradient
        """
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.ls_rate = ls_rate
        self.ls_max_steps = ls_max_steps
        self.elitism_count = max(1, int(pop_size * elitism_ratio))
        self.time_limit = time_limit
        self.fitness_mode = fitness_mode

    def solve(self, cylinders):
        global_best_score = -float('inf')
        global_best_path = None
        
        start_time = time.time()
        epochs = 0
        
        while time.time() - start_time < self.time_limit:
            score, path = memetic_algorithm_core(
                cylinders, 
                pop_size=self.pop_size, 
                generations=self.generations, 
                tournament_size=self.tournament_size, 
                mutation_rate=self.mutation_rate,
                ls_rate=self.ls_rate,
                ls_max_steps=self.ls_max_steps,
                elitism_count=self.elitism_count,
                fitness_mode=self.fitness_mode
            )
            
            if score > global_best_score:
                global_best_score = score
                global_best_path = path.copy()
                
            epochs += 1
            
        total_gens = epochs * self.generations
        print(f"{epochs} écosystèmes ({total_gens} générations) simulés")
        
        return global_best_path.tolist(), global_best_score