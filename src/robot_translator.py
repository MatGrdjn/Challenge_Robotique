import math
import os

class RobotTranslator:
    """
    Convertit une série de coordonnées (x, y) en instructions (angle, avnacement) pour le robot
    """
    def __init__(self, start_x=0.0, start_y=0.0, start_angle=0.0):
        self.x = start_x
        self.y = start_y
        self.angle = start_angle  # En radians
        self.instructions = []

    def move_to(self, target_x, target_y):
        """
        Génère les instructions pour aller de la position courante à (target_x, target_y)
        """
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)

        if distance < 1e-6:
            return 

        # calcul de l'angle cible
        target_angle = math.atan2(dy, dx)

        # calcul de la rotation relative
        rotation = target_angle - self.angle

        # on normalise la rotation entre -pi et pi pour tourner du côté le plus court
        rotation = (rotation + math.pi) % (2 * math.pi) - math.pi

        self.instructions.append(f"TURN {math.degrees(rotation):.5f}")
        self.instructions.append(f"GO {distance:.5f}")

        self.x = target_x
        self.y = target_y
        self.angle = target_angle

    def load_path(self, path_coordinates):
        """
        Charge une liste de tuples (x, y) et génère toutes les instructions
        """
        for (x, y) in path_coordinates:
            self.move_to(x, y)

    def export_to_txt(self, filepath="results/robot_script.txt"):
        """
        Exporte les instructions générées dans un fichier texte
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for instruction in self.instructions:
                f.write(instruction + "\n")
            f.write("FINISH")
        
        print(f"Script exporté avec succès ({len(self.instructions)} instructions) vers : '{filepath}'")