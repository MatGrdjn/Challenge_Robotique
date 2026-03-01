import numpy as np
from .base_solver import BaseSolver

class NearestSolver(BaseSolver):
    """
    Solveur Glouton : Se dirige systématiquement vers le cylindre non visité 
    le plus proche géométriquement, sans aucune considération pour la masse ou les points.
    """
    def __init__(self, current_position=[0.0, 0.0], fitness_mode=0):
        self.fitness_mode = fitness_mode
        self.position = np.array(current_position, dtype=np.float64)

    def solve(self, cylinders):
        n = len(cylinders)
        visited = np.zeros(n, dtype=bool)
        res_indices = []
        
        curr_pos = self.position.copy()
        
        for _ in range(n):
            best_idx = -1
            best_dist = float('inf')
            
            for i in range(n):
                if not visited[i]:
                    dist_sq = np.sum((cylinders[i, :2] - curr_pos)**2)
                    
                    if dist_sq < best_dist:
                        best_dist = dist_sq
                        best_idx = i
                        
            visited[best_idx] = True
            res_indices.append(best_idx)
            
            curr_pos = cylinders[best_idx, :2].copy()
            
        return res_indices, 0.0