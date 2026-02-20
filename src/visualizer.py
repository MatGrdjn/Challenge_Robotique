import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class RouteVisualizer:
    @staticmethod
    def _point_segment_distance(x1, y1, x2, y2, x0, y0):
        """Réplique Python pure de la fonction de distance pour le tracé"""
        dx, dy = x2 - x1, y2 - y1
        if dx == 0.0 and dy == 0.0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        t = max(0.0, min(1.0, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)))
        proj_x, proj_y = x1 + t * dx, y1 + t * dy
        return np.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)

    @classmethod
    def plot_trajectory(cls, cylinders, path, R_col=0.45, save_path="results/trajectory.png"):
        n = len(cylinders)
        status = ['missed'] * n
        visited = set()
        
        curr_x, curr_y = 0.0, 0.0
        trajectory_coords = [(curr_x, curr_y)]
        

        for target_idx in path:
            if target_idx in visited:
                continue
                
            target_x, target_y = cylinders[target_idx, 0], cylinders[target_idx, 1]
            

            for i in range(n):
                if i not in visited and i != target_idx:
                    cx, cy = cylinders[i, 0], cylinders[i, 1]
                    dist = cls._point_segment_distance(curr_x, curr_y, target_x, target_y, cx, cy)
                    if dist <= R_col:
                        status[i] = 'swept'
                        visited.add(i)
                        

            status[target_idx] = 'targeted'
            visited.add(target_idx)
            
            curr_x, curr_y = target_x, target_y
            trajectory_coords.append((curr_x, curr_y))

   
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title("Trajectoire du Robot et Balayage (Sweep)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Coordonnée X")
        ax.set_ylabel("Coordonnée Y")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal') 

        
        
        xs = [p[0] for p in trajectory_coords]
        ys = [p[1] for p in trajectory_coords]
        ax.plot(xs, ys, color='#2c3e50', linewidth=2, linestyle='-', zorder=1, label="Trajectoire principale")
        
        ax.plot(0, 0, marker='s', color='black', markersize=10, zorder=3, label="Départ (0,0)")

        for i in range(n):
            cx, cy, mass, reward = cylinders[i]
            
            if status[i] == 'targeted':
                face_color, edge_color = '#2ecc71', '#27ae60'
                label_prefix = "Cible"
            elif status[i] == 'swept':
                face_color, edge_color = '#f1c40f', '#f39c12'
                label_prefix = "Balayé"
            else:
                face_color, edge_color = '#bdc3c7', '#7f8c8d'
                label_prefix = "Ignoré"

            visual_radius = 0.3 + (mass * 0.1)
            circle = patches.Circle((cx, cy), visual_radius, facecolor=face_color, edgecolor=edge_color, 
                                    linewidth=2, alpha=0.8, zorder=2)
            ax.add_patch(circle)
            
            ax.text(cx, cy, f"#{i}\n{int(reward)}pts", ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='black', zorder=4)

        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='#2c3e50', lw=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f1c40f', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#bdc3c7', markersize=10),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10)
        ]
        ax.legend(custom_lines, ['Trajectoire', 'Cible visée', 'Balayé', 'Ignoré / Non atteint', 'Départ'], 
                  loc='upper right', framealpha=0.9)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualisation sauvegardée dans : '{'save_path}'")
        #plt.show()