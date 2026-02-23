import os
import csv
import time
from datetime import datetime

from solvers.parallel_runner import ParallelRunner
from robot_translator import RobotTranslator
from unity_runner import UnityRunner

class EvaluationPipeline:
    def __init__(self, data_dir, results_dir, unity_exe_path):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.unity_runner = UnityRunner(unity_exe_path)
        self.solvers = []
        
        self.csv_path = os.path.join(self.results_dir, "benchmark_results.csv")
        self._init_csv()

    def _init_csv(self):
        os.makedirs(self.results_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Map", "Algorithme", "Parametres", "Mode_Fitness", "Score_Numba", "Gain_Reel", "Fuel_Reel", "Temps_Reel", "Chemin_Script"])

    def add_solver(self, name, solver_class, params):
        self.solvers.append({
            "name": name,
            "class": solver_class,
            "params": params
        })

    def run_all(self, map_loader_func, target_map=None):

        if target_map:
            map_files = [target_map]
        else:
            map_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
        
        print(f"Début du Pipeline : {len(map_files)} cartes trouvées, {len(self.solvers)} algorithmes à tester.")
        
        for map_file in map_files:
            map_name = os.path.splitext(map_file)[0]
            map_path = os.path.join(self.data_dir, map_file)
            cylinders = map_loader_func(map_path)
            
            print(f"\n=============================================")
            print(f"TRAITEMENT DE LA CARTE : {map_name}")
            print(f"=============================================")
            
            for solver_config in self.solvers:
                algo_name = solver_config["name"]
                params = solver_config["params"]
                
                print(f"\nAlgo : {algo_name}")
                print(f"Params : {params}")
                
                best_path, best_score = ParallelRunner.run(solver_config["class"], params, cylinders)
                
                temp_script = os.path.join(self.results_dir, "temp_script.txt")

                translator = RobotTranslator(cylinders=cylinders)
                translator.generate_script(best_path, temp_script)
                
                real_metrics = self.unity_runner.run_simulation(map_path, temp_script)
                
                if real_metrics is not None:
                    gain_reel, fuel_reel, temps_reel = real_metrics
                    
                    save_dir = os.path.join(self.results_dir, map_name, algo_name)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    final_script_name = f"script_{algo_name}_score_{gain_reel}.txt"
                    final_script_path = os.path.join(save_dir, final_script_name)
                    
                    if os.path.exists(final_script_path):
                        os.remove(final_script_path)
                    os.rename(temp_script, final_script_path)
                    
                    with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            map_name,
                            algo_name,
                            str(params),
                            params.get('fitness_mode', 0),
                            f"{best_score:.2f}",
                            gain_reel,
                            f"{fuel_reel:.2f}",
                            f"{temps_reel:.2f}",
                            final_script_path
                        ])
                    print(f"Résultat sauvegardé dans {final_script_path}")
                else:
                    if os.path.exists(temp_script):
                        os.remove(temp_script)