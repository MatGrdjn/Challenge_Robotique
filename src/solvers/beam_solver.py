from .base_solver import BaseSolver
from utils_solver import beam_search_core

class BeamSearchSolver(BaseSolver):
    def __init__(self, beam_width=2000, fitness_mode=0):

        self.beam_width = beam_width
        self.fitness_mode = fitness_mode

    def solve(self, cylinders):
        print(f"Lancement du Beam Search (Largeur K={self.beam_width}, Mode={self.fitness_mode})")
        
        best_path_array, best_score = beam_search_core(
            cylinders, 
            beam_width=self.beam_width,
            fitness_mode=self.fitness_mode
        )
        
        return best_path_array.tolist(), best_score