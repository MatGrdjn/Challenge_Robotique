import numpy as np
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



@njit(cache=True)
def evaluate_path(path, cylinders, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45):

    n_cylinders = len(cylinders)
    visited = np.zeros(n_cylinders, dtype=np.bool_)
    
    curr_x, curr_y = 0.0, 0.0
    M, T, Q, Reward = 0.0, 0.0, 0.0, 0.0
    
    swept = np.zeros(n_cylinders, dtype=np.int32)
    
    for target_idx in path:
        if visited[target_idx]:
            continue
            
        target_x = cylinders[target_idx, 0]
        target_y = cylinders[target_idx, 1]
        
        dx = target_x - curr_x
        dy = target_y - curr_y
        D = np.sqrt(dx**2 + dy**2)
        
        V = V0 * np.exp(-a * M)
        q_rate = b * M + b0
        
        delta_T = D / V
        delta_Q = q_rate * D
        
        if T + delta_T > Tmax or Q + delta_Q > Qmax:
            break
            
        swept_count = 0
        for i in range(n_cylinders):
            if not visited[i] and i != target_idx:
                cx = cylinders[i, 0]
                cy = cylinders[i, 1]
                dist = point_segment_distance(curr_x, curr_y, target_x, target_y, cx, cy)
                if dist <= R_col:
                    swept[swept_count] = i
                    swept_count += 1
                    
        curr_x = target_x
        curr_y = target_y
        T += delta_T
        Q += delta_Q
        
        visited[target_idx] = True
        M += cylinders[target_idx, 2]
        Reward += cylinders[target_idx, 3]
        
        for i in range(swept_count):
            idx = swept[i]
            visited[idx] = True
            M += cylinders[idx, 2]
            Reward += cylinders[idx, 3]
            
    fitness = (Reward * 1e10) + ((Qmax - Q) * 1e5) + (Tmax - T)
    
    return fitness, Reward, Q, T


@njit(cache=True)
def evaluate_path(path, cylinders, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45):
    """
    Simule le parcours exact
    Visée de centre à centre et téléportation au centre à l'arrivée pour éviter la dérive
    """
    n_cylinders = len(cylinders)
    
    visited = 0 
    
    curr_x, curr_y = 0.0, 0.0
    M, T, Q, Reward = 0.0, 0.0, 0.0, 0.0
    
    swept = np.zeros(n_cylinders, dtype=np.int32)
    
    for target_idx in path:
        # Vérification par masque de bits
        if (visited & (1 << target_idx)):
            continue
            
        target_x = cylinders[target_idx, 0]
        target_y = cylinders[target_idx, 1]
        
        dx = target_x - curr_x
        dy = target_y - curr_y
        D = np.sqrt(dx**2 + dy**2)
        
        V = V0 * np.exp(-a * M)
        q_rate = b * M + b0
        
        delta_T = D / V
        delta_Q = q_rate * D
        
        if T + delta_T > Tmax or Q + delta_Q > Qmax:
            break
            
        swept_count = 0
        for i in range(n_cylinders):
            if not (visited & (1 << i)) and i != target_idx:
                cx = cylinders[i, 0]
                cy = cylinders[i, 1]
                dist = point_segment_distance(curr_x, curr_y, target_x, target_y, cx, cy)
                if dist <= R_col:
                    swept[swept_count] = i
                    swept_count += 1
                    
        curr_x = target_x
        curr_y = target_y
        T += delta_T
        Q += delta_Q
        
        visited |= (1 << target_idx)
        M += cylinders[target_idx, 2]
        Reward += cylinders[target_idx, 3]
        
        for i in range(swept_count):
            idx = swept[i]
            visited |= (1 << idx)
            M += cylinders[idx, 2]
            Reward += cylinders[idx, 3]
            
    fitness = (Reward * 1e10) + ((Qmax - Q) * 1e5) + (Tmax - T)
    return fitness, Reward, Q, T


@njit(cache=True)
def fast_random_rollout(path_prefix, cylinders, V0=1.0, a=0.0698, b=3.0, b0=100.0, Tmax=600.0, Qmax=10000.0, R_col=0.45):
    """
    Prend un début de chemin et le complète avec les cylindres restants 
    mélangés aléatoirement et renvoie le score exact
    """
    n = len(cylinders)
    full_path = np.full(n, -1, dtype=np.int32)
    
    visited = 0
    prefix_len = len(path_prefix)
    
    for i in range(prefix_len):
        full_path[i] = path_prefix[i]
        visited |= (1 << path_prefix[i])
        
    remaining = np.zeros(n - prefix_len, dtype=np.int32)
    idx = 0
    for i in range(n):
        if not (visited & (1 << i)):
            remaining[idx] = i
            idx += 1
            
    np.random.shuffle(remaining)
    
    for i in range(len(remaining)):
        full_path[prefix_len + i] = remaining[i]
        
    fitness = evaluate_path(full_path, cylinders, V0, a, b, b0, Tmax, Qmax, R_col)[0]
    return fitness, full_path