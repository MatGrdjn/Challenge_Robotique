import numpy as np
from .base_solver import BaseSolver

class WeightedRatioSolver(BaseSolver):
    def __init__(self, current_position=[0.0, 0.0], wp=1.0, wd=1.0, wm=0.0, fitness_mode=0):
        self.position = np.array(current_position, dtype=np.float64)
        self.wp = wp
        self.wd = wd
        self.wm = wm
        self.fitness_mode = fitness_mode

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
                mass = cylinders[i, 2]
                dist = np.linalg.norm(cylinders[i, :2] - curr_pos)

                numerator = points ** self.wp
                denominator = ((dist + 1e-6) ** self.wd) * ((mass + 1e-6) ** self.wm)
                
                ratio = numerator / denominator
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
                    
            res_indices.append(best_idx)
            visited[best_idx] = True
            curr_pos = cylinders[best_idx, :2].copy()
            
        return res_indices, 0.0