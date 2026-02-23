import os
import time
import shutil
import subprocess
import re

class UnityRunner:
    def __init__(self, exe_path, challenge_dir="C://challenge"):
        self.exe_path = exe_path
        self.challenge_dir = challenge_dir
        
        self.map_dest = os.path.join(self.challenge_dir, "donnees-map.txt")
        self.script_dest = os.path.join(self.challenge_dir, "script.txt")
        self.score_dest = os.path.join(self.challenge_dir, "score.txt")
        
        os.makedirs(self.challenge_dir, exist_ok=True)

    def run_simulation(self, map_source, script_source, timeout=610):

        shutil.copyfile(map_source, self.map_dest)
        shutil.copyfile(script_source, self.script_dest)
        
        if os.path.exists(self.score_dest):
            os.remove(self.score_dest)
            
        print("Lancement de la simulation Unity")
        process = subprocess.Popen([self.exe_path])
        
        start_time = time.time()
        score_data = None
        
        while time.time() - start_time < timeout:
            if os.path.exists(self.score_dest):
                try:
                    with open(self.score_dest, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            score_data = self._parse_score(content)
                            if score_data:
                                break 
                except IOError:
                    pass
            
            time.sleep(1.0)
            
        try:
            process.kill()
        except:
            pass
            
        if score_data is None:
            print("Simulation échouée ou Timeout atteint.")
        else:
            print(f"Simulation terminée : Gain={score_data[0]}, Fuel={score_data[1]:.1f}, Temps={score_data[2]:.1f}")
            
        return score_data

    def _parse_score(self, content):

        content_normalized = content.replace(',', '.')
        
        match = re.search(r"gain\s*=\s*([\d\.]+)\s*fuel\s*=\s*([\d\.]+)\s*temps\s*=\s*([\d\.]+)", content_normalized)
        
        if match:
            gain = int(float(match.group(1)))
            fuel = float(match.group(2))
            temps = float(match.group(3))
            return gain, fuel, temps
        return None