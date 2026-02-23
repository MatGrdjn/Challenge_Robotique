import os
import csv
import re
import json
from datetime import datetime

class ResultTracker:
    """
    Parse les fichiers de sortie du simulateur et archive les résultats
    dans un fichier CSV global pour suivre les performances
    """
    def __init__(self, csv_filepath="results/experiment_history.csv"):
        self.csv_filepath = csv_filepath
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)
        
        if not os.path.isfile(self.csv_filepath):
            with open(self.csv_filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    "Date", 
                    "Algorithme", 
                    "Parametres", 
                    "Temps_Exec_sec", 
                    "Fitness_Calculee", 
                    "Sim_Gain", 
                    "Sim_Fuel", 
                    "Sim_Temps"
                ])

    def parse_score_file(self, score_filepath="score.txt"):
        """
        Lit le fichier texte généré par le simulateur et extrait les valeurs
        """
        if not os.path.exists(score_filepath):
            print(f"Fichier {score_filepath} introuvable")
            return None, None, None

        with open(score_filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        content = content.replace(',', '.')

        match_gain = re.search(r'gain\s*=\s*([0-9.]+)', content, re.IGNORECASE)
        match_fuel = re.search(r'fuel\s*=\s*([0-9.]+)', content, re.IGNORECASE)
        match_temps = re.search(r'temps\s*=\s*([0-9.]+)', content, re.IGNORECASE)

        gain = float(match_gain.group(1)) if match_gain else 0.0
        fuel = float(match_fuel.group(1)) if match_fuel else 0.0
        temps = float(match_temps.group(1)) if match_temps else 0.0

        return gain, fuel, temps

    def log_experiment(self, algo_name, params_dict, exec_time, calc_fitness, score_filepath="score.txt"):
        """
        Parse le score et ajoute une nouvelle ligne dans le fichier CSV
        """
        sim_gain, sim_fuel, sim_temps = self.parse_score_file(score_filepath)
        
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        params_str = json.dumps(params_dict)

        with open(self.csv_filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                current_date,
                algo_name,
                params_str,
                round(exec_time, 3),
                round(calc_fitness, 2),
                sim_gain,
                sim_fuel,
                sim_temps
            ])
            
        print(f"Résultats archivés dans {self.csv_filepath}")