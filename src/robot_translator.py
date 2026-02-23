import math
import os

class RobotTranslator:
    """
    Traducteur intelligent qui simule exactement la physique pour éviter la désynchronisation des angles après une collision de balayage
    """
    def __init__(self, cylinders, R_col=0.45, start_x=0.0, start_y=0.0, start_angle=0.0):
        self.cylinders = cylinders
        self.R_col_sq = R_col ** 2
        self.x = start_x
        self.y = start_y
        self.angle = start_angle
        self.instructions = []
        self.visited = 0

    def generate_script(self, best_path, filepath="results/script_robot.txt"):
        """
        Lit l'ordre des cibles intentionnelles, simule les collisions,
        et génère les vraies commandes synchronisées
        """
        for target_idx in best_path:
            while not (self.visited & (1 << target_idx)):
                target_x = self.cylinders[target_idx, 0]
                target_y = self.cylinders[target_idx, 1]
                
                dx = target_x - self.x
                dy = target_y - self.y
                D = math.hypot(dx, dy)
                
                if D < 1e-6:
                    self.visited |= (1 << target_idx)
                    break
                    
                target_angle = math.atan2(dy, dx)
                rotation = target_angle - self.angle
                
                rotation = (rotation + math.pi) % (2 * math.pi) - math.pi
                
                self.instructions.append(f"TURN {math.degrees(rotation):.5f}")
                self.instructions.append(f"GO {D:.5f}")
                
                hit_idx = target_idx
                min_t = D
                
                for i in range(len(self.cylinders)):
                    if self.visited & (1 << i): continue
                    if i == target_idx: continue
                    
                    cx, cy = self.cylinders[i, 0], self.cylinders[i, 1]
                    vx, vy = cx - self.x, cy - self.y
                    dot = vx * dx + vy * dy
                    
                    if dot <= 0: continue
                    
                    t = dot / D
                    if t >= min_t: continue
                    
                    d_sq = (vx**2 + vy**2) - t**2
                    if d_sq <= self.R_col_sq:
                        dist_to_hit = t - math.sqrt(abs(self.R_col_sq - d_sq))
                        if dist_to_hit < min_t:
                            min_t = dist_to_hit
                            hit_idx = i
                            
                self.x = self.cylinders[hit_idx, 0]
                self.y = self.cylinders[hit_idx, 1]
                
                self.angle = target_angle 
                self.visited |= (1 << hit_idx)

        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)


        with open(filepath, 'w', encoding='utf-8') as f:
            for instruction in self.instructions:
                f.write(instruction + "\n")
            f.write("FINISH")
                
        print(f"Script généré avec succès ({len(self.instructions)} instructions) vers : {filepath}")