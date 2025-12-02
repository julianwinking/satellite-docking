import os
import logging
from pathlib import Path
from satellite_docking.simulation.ex13 import ex13_evaluation
from satellite_docking.simulation.utils_config import sim_context_from_yaml
from satellite_docking.simulation.get_config import get_config
from satellite_docking.simulation.utils_output import out_dir

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Get output directory
    out_path = out_dir("13")
    os.makedirs(out_path, exist_ok=True)
    
    # Get configs
    # Assuming main.py is in src/
    config_dir = Path(__file__).parent / "satellite_docking/simulation"
    configs = get_config()
    
    for c in configs:
        config_file = config_dir / c
        print(f"Running simulation for {c}...")
        
        if not config_file.exists():
            print(f"Config file not found: {config_file}")
            continue

        try:
            sim_context = sim_context_from_yaml(str(config_file))
            res = ex13_evaluation(sim_context)
            
            print(f"Score: {res[1]}")
            
        except Exception as e:
            print(f"Failed to run {c}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
