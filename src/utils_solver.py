import numpy as np
import math
from numba import njit

@njit(cache=True)
def point_segment_distance(x1, y1, x2, y2, x0, y0):
    """
    Calcule la distance orthogonale entre le segment [A(x1,y1), B(x2,y2)] et le point C(x0,y0)
    Sert à vérifier si le cylindre C est balayé pendant le trajet
    """
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0.0 and dy == 0.0:
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)
    t = max(0.0, min(1.0, t))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return np.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)



@njit(cache=True, fastmath=True)
def evaluate_path(path, cylinders, fitness_mode=0, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45):
    """
    Simule le parcours exact avec interruption de collision
    """
    visited = 0
    curr_x, curr_y = 0.0, 0.0
    M, T, Q, Reward = 0.0, 0.0, 0.0, 0.0
    
    R_col_sq = R_col * R_col
    
    for p_idx in range(20):
        target_idx = path[p_idx]
        
        while not (visited & (1 << target_idx)):
            target_x = cylinders[target_idx, 0]
            target_y = cylinders[target_idx, 1]
            
            dx = target_x - curr_x
            dy = target_y - curr_y
            D = math.sqrt(dx**2 + dy**2)
            

            if D < 1e-6:
                visited |= (1 << target_idx)
                M += cylinders[target_idx, 2]
                Reward += cylinders[target_idx, 3]
                break

            hit_idx = target_idx
            min_t = D 
            
            for i in range(20):
                if visited & (1 << i): 
                    continue
                if i == target_idx: 
                    continue
                
                cx = cylinders[i, 0]
                cy = cylinders[i, 1]
                vx = cx - curr_x
                vy = cy - curr_y
                
                dot = vx * dx + vy * dy
                if dot <= 0: 
                    continue
                
                t = dot / max(D, 1e-12)
                
                if t - R_col >= min_t: 
                    continue 
                
                d_sq = (vx**2 + vy**2) - t**2
                
                if d_sq <= R_col_sq:
                    dist_to_hit = t - math.sqrt(abs(R_col_sq - d_sq))
                    if -1e-5 < dist_to_hit < min_t:
                        min_t = dist_to_hit
                        hit_idx = i

            actual_target_x = cylinders[hit_idx, 0]
            actual_target_y = cylinders[hit_idx, 1]
            
            real_dx = actual_target_x - curr_x
            real_dy = actual_target_y - curr_y
            real_D = math.sqrt(real_dx**2 + real_dy**2)
            
            V = max(V0 * math.exp(-a * M), 1e-9)
            q_rate = b * M + b0
            
            delta_T = real_D / V
            delta_Q = q_rate * real_D
            
            if fitness_mode == 0:
                if T + delta_T > Tmax or Q + delta_Q > Qmax:
                    ratio_Q = (Qmax - Q) / delta_Q if delta_Q > 0 else 0
                    ratio_T = (Tmax - T) / delta_T if delta_T > 0 else 0
                    ratio = min(ratio_Q, ratio_T)
                    
                    ratio = max(0.0, min(1.0, ratio))
                    
                    fitness = (Reward * 1e10) + (ratio * 1e7) + (Qmax - Q)
                    return fitness, Reward, Q, T
            
            curr_x = actual_target_x
            curr_y = actual_target_y
            T += delta_T
            Q += delta_Q
            
            visited |= (1 << hit_idx)
            M += cylinders[hit_idx, 2]
            Reward += cylinders[hit_idx, 3]

    if fitness_mode == 0:
        fitness = (Reward * 1e10) + 1e7 + ((Qmax - Q) * 1e5) + (Tmax - T)
    else:
        fitness = -(Q * 1e5) - T

    return fitness, Reward, Q, T



@njit(cache=True)
def set_numba_seed(seed):
    """Initialise le moteur aléatoire interne de Numba"""
    np.random.seed(seed)


@njit(cache=True)
def fast_random_rollout(prefix_array, prefix_len, full_path, cylinders, fitness_mode=0, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45):
    """
    Prend un début de chemin et le complète avec les cylindres restants 
    mélangés aléatoirement et renvoie le score exact
    """
    visited = 0
    
    for i in range(prefix_len):
        full_path[i] = prefix_array[i]
        visited |= (1 << prefix_array[i])
        
    remaining_count = 0
    for i in range(20):
        if not (visited & (1 << i)):
            full_path[prefix_len + remaining_count] = i
            remaining_count += 1
            
    # Mélange Fisher-Yates
    for i in range(remaining_count - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = full_path[prefix_len + i]
        full_path[prefix_len + i] = full_path[prefix_len + j]
        full_path[prefix_len + j] = tmp
        
    fit, _, _, _ = evaluate_path(full_path, cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)
    return fit


@njit(cache=True, fastmath=True)
def simulated_annealing_core(cylinders, fitness_mode=0,  T_init=10000.0, T_final=0.1, alpha=0.9999, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45):
    """
    Recuit Simulé
    """

    current_path = np.empty(20, dtype=np.int32)
    for i in range(20):
        current_path[i] = i
    
    for i in range(19, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = current_path[i]
        current_path[i] = current_path[j]
        current_path[j] = tmp
        
    current_score = evaluate_path(current_path, cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
    
    best_path = current_path.copy()
    best_score = current_score
    
    T = T_init
    new_path = np.empty(20, dtype=np.int32)
    
    if fitness_mode == 0:
        scale_factor = 1e8
    else:
        scale_factor = 1e5
    
    while T > T_final:
        # Copie in-place
        for i in range(20):
            new_path[i] = current_path[i]
            
        # mutation (2-opt Swap on inverse un sous-segment)
        idx1 = np.random.randint(0, 19)
        idx2 = np.random.randint(idx1 + 1, 20)
        
        left = idx1
        right = idx2
        while left < right:
            tmp = new_path[left]
            new_path[left] = new_path[right]
            new_path[right] = tmp
            left += 1
            right -= 1
            
        # évaluation
        new_score = evaluate_path(new_path, cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
        
        delta = new_score - current_score
        
        # critère de Metropolis
        if delta > 0:
    
            for i in range(20):
                current_path[i] = new_path[i]
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                for i in range(20):
                    best_path[i] = current_path[i]
        else:

            prob = math.exp(delta / (T * scale_factor))
            if np.random.rand() < prob:
                for i in range(20):
                    current_path[i] = new_path[i]
                current_score = new_score
                
        # refroidissement
        T *= alpha
        
    return best_score, best_path



@njit(cache=True, fastmath=True)
def tournament_selection(fitnesses, pop_size, tournament_size):
    """Sélectionne le meilleur individu parmi un sous-groupe aléatoire"""
    best_idx = np.random.randint(0, pop_size)
    best_fit = fitnesses[best_idx]
    
    for _ in range(tournament_size - 1):
        idx = np.random.randint(0, pop_size)
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_idx = idx
            
    return best_idx

@njit(cache=True, fastmath=True)
def ox_crossover(p1, p2, child):
    """
    Order Crossover (OX1)
    Préserve un segment de P1 et complète avec l'ordre relatif de P2
    """
    a = np.random.randint(0, 19)
    b = np.random.randint(a + 1, 20)
    
    visited = 0
    for i in range(a, b):
        child[i] = p1[i]
        visited |= (1 << p1[i])
        
    idx_child = b
    idx_p2 = b
    
    for _ in range(20):
        val = p2[idx_p2 % 20]
        if not (visited & (1 << val)):
            child[idx_child % 20] = val
            idx_child += 1
        idx_p2 += 1

@njit(cache=True, fastmath=True)
def mutate_2opt_inplace(ind, mutation_rate):
    """Mutation par inversion de segment (2-Opt Swap)"""
    if np.random.rand() < mutation_rate:
        idx1 = np.random.randint(0, 19)
        idx2 = np.random.randint(idx1 + 1, 20)
        
        while idx1 < idx2:
            tmp = ind[idx1]
            ind[idx1] = ind[idx2]
            ind[idx2] = tmp
            idx1 += 1
            idx2 -= 1

@njit(cache=True, fastmath=True)
def genetic_algorithm_core(cylinders, pop_size, generations, tournament_size, mutation_rate, elitism_count, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45, fitness_mode=0):
    """
    Le moteur complet de l'Algorithme Génétique
    """
    population = np.empty((pop_size, 20), dtype=np.int32)
    new_population = np.empty((pop_size, 20), dtype=np.int32)
    fitnesses = np.empty(pop_size, dtype=np.float64)
    
    for i in range(pop_size):
        for j in range(20):
            population[i, j] = j
        for j in range(19, 0, -1):
            k = np.random.randint(0, j + 1)
            tmp = population[i, j]
            population[i, j] = population[i, k]
            population[i, k] = tmp
            
    for i in range(pop_size):
        fitnesses[i] = evaluate_path(population[i], cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
        
    best_overall_score = -np.inf
    best_overall_path = np.empty(20, dtype=np.int32)
    
    for gen in range(generations):
        order = np.argsort(fitnesses)[::-1]
        
        if fitnesses[order[0]] > best_overall_score:
            best_overall_score = fitnesses[order[0]]
            for j in range(20):
                best_overall_path[j] = population[order[0], j]
                
        for i in range(elitism_count):
            for j in range(20):
                new_population[i, j] = population[order[i], j]
                
        #reproduction 
        for i in range(elitism_count, pop_size):
            p1_idx = tournament_selection(fitnesses, pop_size, tournament_size)
            p2_idx = tournament_selection(fitnesses, pop_size, tournament_size)
            
            ox_crossover(population[p1_idx], population[p2_idx], new_population[i])
            mutate_2opt_inplace(new_population[i], mutation_rate)
            
        # Remplacement et évaluation
        for i in range(pop_size):
            for j in range(20):
                population[i, j] = new_population[i, j]
                
            if i >= elitism_count:
                fitnesses[i] = evaluate_path(population[i], cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
                
    return best_overall_score, best_overall_path



@njit(cache=True, fastmath=True)
def beam_search_core(cylinders, beam_width, fitness_mode=0, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45):
    """
    Implémentation haute performance du Beam Search
    """
    n_cylinders = 20
    
    paths_even = np.full((beam_width, n_cylinders), -1, dtype=np.int32)
    scores_even = np.full(beam_width, -np.inf, dtype=np.float64)
    
    paths_odd = np.full((beam_width, n_cylinders), -1, dtype=np.int32)
    scores_odd = np.full(beam_width, -np.inf, dtype=np.float64)
    
    initial_candidates = min(n_cylinders, beam_width)
    for i in range(initial_candidates):
        paths_even[i, 0] = i
        scores_even[i] = evaluate_path(paths_even[i], cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
        
    current_beam_size = initial_candidates

    for level in range(1, n_cylinders):
        if level % 2 != 0:
            parent_paths = paths_even
            parent_scores = scores_even
            child_paths = paths_odd
            child_scores = scores_odd
        else:
            parent_paths = paths_odd
            parent_scores = scores_odd
            child_paths = paths_even
            child_scores = scores_even
            
        child_scores[:] = -np.inf
        child_paths[:] = -1
        
        max_candidates = beam_width * n_cylinders
        candidate_paths = np.full((max_candidates, n_cylinders), -1, dtype=np.int32)
        candidate_scores = np.full(max_candidates, -np.inf, dtype=np.float64)
        cand_count = 0
        
        for i in range(current_beam_size):
            if parent_paths[i, 0] == -1 or parent_scores[i] == -np.inf: continue

            visited_mask = 0
            for k in range(level):
                visited_mask |= (1 << parent_paths[i, k])
                
            for target_idx in range(n_cylinders):
                if not (visited_mask & (1 << target_idx)):
                    for k in range(level):
                        candidate_paths[cand_count, k] = parent_paths[i, k]
                    candidate_paths[cand_count, level] = target_idx
                    
                    score = evaluate_path(candidate_paths[cand_count], cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
                    candidate_scores[cand_count] = score
                    cand_count += 1
        
        valid_scores = candidate_scores[:cand_count]
        sorted_indices = np.argsort(valid_scores)[::-1]
        
        current_beam_size = min(cand_count, beam_width)
        for k in range(current_beam_size):
            best_idx = sorted_indices[k]
            child_scores[k] = candidate_scores[best_idx]
            for j in range(n_cylinders):
                child_paths[k, j] = candidate_paths[best_idx, j]

    best_score = child_scores[0]
    best_path = child_paths[0]
            
    return best_path, best_score



@njit(cache=True, fastmath=True)
def fast_local_search_2opt(path, cylinders, fitness_mode, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45, max_steps=50):

    best_score = evaluate_path(path, cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
    improved = True
    steps = 0
    
    while improved and steps < max_steps:
        improved = False
        
        for i in range(19):
            for j in range(i + 1, 20):
                left, right = i, j
                while left < right:
                    tmp = path[left]
                    path[left] = path[right]
                    path[right] = tmp
                    left += 1
                    right -= 1
                    
                new_score = evaluate_path(path, cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
                
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    break 
                else:
                    left, right = i, j
                    while left < right:
                        tmp = path[left]
                        path[left] = path[right]
                        path[right] = tmp
                        left += 1
                        right -= 1
            if improved:
                break
                
        steps += 1
        
    return best_score

@njit(cache=True, fastmath=True)
def memetic_algorithm_core(cylinders, pop_size, generations, tournament_size, mutation_rate, ls_rate, ls_max_steps, elitism_count, fitness_mode=0, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45):

    population = np.empty((pop_size, 20), dtype=np.int32)
    new_population = np.empty((pop_size, 20), dtype=np.int32)
    
    fitnesses = np.empty(pop_size, dtype=np.float64)
    new_fitnesses = np.empty(pop_size, dtype=np.float64)
    
    for i in range(pop_size):
        for j in range(20):
            population[i, j] = j
        for j in range(19, 0, -1):
            k = np.random.randint(0, j + 1)
            tmp = population[i, j]
            population[i, j] = population[i, k]
            population[i, k] = tmp
            
    for i in range(pop_size):
        fitnesses[i] = fast_local_search_2opt(population[i], cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col, ls_max_steps)
        
    best_overall_score = -np.inf
    best_overall_path = np.empty(20, dtype=np.int32)
    
    for gen in range(generations):
        order = np.argsort(fitnesses)[::-1]
        
        if fitnesses[order[0]] > best_overall_score:
            best_overall_score = fitnesses[order[0]]
            for j in range(20):
                best_overall_path[j] = population[order[0], j]
                
        for i in range(elitism_count):
            for j in range(20):
                new_population[i, j] = population[order[i], j]
            new_fitnesses[i] = fitnesses[order[i]]
                
        for i in range(elitism_count, pop_size):
            p1_idx = tournament_selection(fitnesses, pop_size, tournament_size)
            p2_idx = tournament_selection(fitnesses, pop_size, tournament_size)
            
            ox_crossover(population[p1_idx], population[p2_idx], new_population[i])
            
            mutate_2opt_inplace(new_population[i], mutation_rate)
            
            if np.random.rand() < ls_rate:
                new_fitnesses[i] = fast_local_search_2opt(new_population[i], cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col, ls_max_steps)
            else:
                new_fitnesses[i] = evaluate_path(new_population[i], cylinders, fitness_mode, V0, a, b, b0, Tmax, Qmax, R_col)[0]
                
        for i in range(pop_size):
            for j in range(20):
                population[i, j] = new_population[i, j]
            fitnesses[i] = new_fitnesses[i]

    return best_overall_score, best_overall_path