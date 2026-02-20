import os
import time
import numpy as np
import concurrent.futures

def _worker_task(solver_class, solver_kwargs, cylinders, seed):
    """
    Fonction isolée exécutée par chaque coeur
    Le seed unique garantit que chaque MCTS explore des branches différentes
    """
    np.random.seed(seed)
    solver = solver_class(**solver_kwargs)
    return solver.solve(cylinders)

class ParallelRunner:
    
    @staticmethod
    def run(solver_class, solver_kwargs, cylinders, n_cores=None):
        if n_cores is None:
            n_cores = os.cpu_count() or 4
            
        print(f"Déploiement de {solver_class.__name__} sur {n_cores} coeurs")
        
        all_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            # Création des tâches avec des seeds distincts
            futures = [
                executor.submit(
                    _worker_task, 
                    solver_class, 
                    solver_kwargs, 
                    cylinders, 
                    int(time.time() * 1000) % (i + 12345)
                )
                for i in range(n_cores)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                all_results.append(future.result())
                
        # Récupération du champion absolu
        global_best_score = -float('inf')
        global_best_path = None
        
        for path, score in all_results:
            if score > global_best_score:
                global_best_score = score
                global_best_path = path
                
        return global_best_path, global_best_score