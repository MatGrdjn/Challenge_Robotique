import time
from .base_solver import BaseSolver
from utils_solver import genetic_algorithm_core


class GASolver(BaseSolver):
    def __init__(self, pop_size=200, generations=1000, tournament_size=5, mutation_rate=0.2, elitism_ratio=0.05, time_limit=900.0, fitness_mode=0):
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.elitism_count = max(1, int(pop_size * elitism_ratio))
        self.time_limit = time_limit
        self.fitness_mode = fitness_mode

    def solve(self, cylinders):
        global_best_score = -float('inf')
        global_best_path = None
        
        start_time = time.time()
        epochs = 0
        
        while time.time() - start_time < self.time_limit:
            score, path = genetic_algorithm_core(
                cylinders,
                fitness_mode=self.fitness_mode, 
                pop_size=self.pop_size, 
                generations=self.generations, 
                tournament_size=self.tournament_size, 
                mutation_rate=self.mutation_rate, 
                elitism_count=self.elitism_count
            )
            
            if score > global_best_score:
                global_best_score = score
                global_best_path = path.copy()
                
            epochs += 1
            
        total_gens = epochs * self.generations
        print(f"{epochs} populations simulées ({total_gens} générations) sur ce coeur")
        return global_best_path.tolist(), global_best_score