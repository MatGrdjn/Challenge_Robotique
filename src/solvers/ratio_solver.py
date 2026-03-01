import numpy as np
from .base_solver import BaseSolver

class RatioSolver(BaseSolver):
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
            best_ratio = -float('inf')
            
            for i in range(n):
                if visited[i]:
                    continue
                    
                points = cylinders[i, 3]
                
                dist = np.linalg.norm(cylinders[i, :2] - curr_pos)
                
                ratio = points / (dist + 1e-6)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
                    
            res_indices.append(best_idx)
            visited[best_idx] = True
            
            curr_pos = cylinders[best_idx, :2].copy()
            
        return res_indices, 0.0