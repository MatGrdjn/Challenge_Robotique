import math
import time
import numpy as np
from .base_solver import BaseSolver
from utils_solver import fast_random_rollout

class MCTSSolver(BaseSolver):
    def __init__(self, iterations=100000, exploration_constant=1.414, time_limit=None):
        """
        :param iterations: Nombre de simulations (rollouts) à effectuer.
        :param exploration_constant: Constante C (généralement sqrt(2) = 1.414)
        :param time_limit: Temps maximum alloué en secondes (optionnel)
        """
        self.iterations = iterations
        self.C = exploration_constant
        self.time_limit = time_limit
        
        self.tree = {}
        self.n_cylinders = 0
        
        self.global_min_score = float('inf')
        self.global_max_score = -float('inf')

    def solve(self, cylinders):
        self.n_cylinders = len(cylinders)
        root_state = ()
        
        self.tree[root_state] = [0, 0.0, list(range(self.n_cylinders))]
        
        best_overall_score = -float('inf')
        best_overall_path = None
        
        start_time = time.time()
        
        for i in range(self.iterations):
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
                
            #Selection
            node_state = root_state
            path_list = list(node_state)

            while not self.tree[node_state][2] and len(node_state) < self.n_cylinders:
                node_state = self._select_best_child(node_state)
                path_list = list(node_state)
                
            #Expansion
            if len(node_state) < self.n_cylinders and self.tree[node_state][2]:
                idx = np.random.randint(len(self.tree[node_state][2]))
                action = self.tree[node_state][2].pop(idx)
                
                path_list.append(action)
                node_state = tuple(path_list)
                
                untried = [c for c in range(self.n_cylinders) if c not in node_state]
                self.tree[node_state] = [0, 0.0, untried]
                
            #Rollout
            prefix_array = np.array(path_list, dtype=np.int32)
            score, full_path = fast_random_rollout(prefix_array, cylinders)
            
            if score < self.global_min_score:
                self.global_min_score = score
            if score > self.global_max_score:
                self.global_max_score = score
                
            if score > best_overall_score:
                best_overall_score = score
                best_overall_path = full_path.copy()


            #Backpropagation
            current_backprop = path_list.copy()
            state_tuple = tuple(current_backprop)
            
            while True:
                # màj du noeud
                self.tree[state_tuple][0] += 1  # N += 1
                self.tree[state_tuple][1] += score  # Q += score
                
                if len(state_tuple) == 0:
                    break
                current_backprop.pop()
                state_tuple = tuple(current_backprop)

        elapsed = time.time() - start_time
        print(f"MCTS terminé: {i+1} itérations en {elapsed:.2f}s")
        print(f"Meilleur score trouvé : {best_overall_score:,.2f}")
        
        return best_overall_path.tolist(), best_overall_score

    def _select_best_child(self, state):
        """Sélectionne l'enfant maximisant la formule UCB1 avec la normalisation"""
        best_score = -float('inf')
        best_child = None
        
        parent_visits = self.tree[state][0]
        
        for action in range(self.n_cylinders):
            if action in state:
                continue
                
            child_state = state + (action,)
            if child_state in self.tree:
                child_node = self.tree[child_state]
                n_visits = child_node[0]
                q_value = child_node[1]
                
                if n_visits == 0:
                    return child_state
                    
                average_reward = q_value / n_visits
                
                # normalisation [0, 1]
                if self.global_max_score > self.global_min_score:
                    normalized_reward = (average_reward - self.global_min_score) / (self.global_max_score - self.global_min_score)
                else:
                    normalized_reward = 0.5
                
                # formule UCB1
                exploration = self.C * math.sqrt(math.log(parent_visits) / n_visits)
                uct_score = normalized_reward + exploration
                
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child_state
                    
        return best_child