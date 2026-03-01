import numpy as np
from .base_solver import BaseSolver

class SimpleSolver(BaseSolver):
    def __init__(self, current_position=[0.0, 0.0], rule="lightest_first", fitness_mode=0):
        self.rule = rule
        self.fitness_mode = fitness_mode
        self.position = np.array(current_position, dtype=np.float64)

    def solve(self, cylinders):
        n = len(cylinders)
        visited = np.zeros(n, dtype=bool)
        res_indices = []
        
        curr_pos = self.position.copy()
        
        for target_mass in [1.0, 2.0, 3.0]:
            
            while True:
                best_idx = -1
                best_dist = float('inf')
                
                for i in range(n):
                    if not visited[i] and cylinders[i, 2] == target_mass:
                        dist_sq = np.sum((cylinders[i, :2] - curr_pos)**2)
                        
                        if dist_sq < best_dist:
                            best_dist = dist_sq
                            best_idx = i
                            
                if best_idx == -1:
                    break
                    
                visited[best_idx] = True
                res_indices.append(best_idx)
                curr_pos = cylinders[best_idx, :2].copy()
                
        return res_indices, 0.0