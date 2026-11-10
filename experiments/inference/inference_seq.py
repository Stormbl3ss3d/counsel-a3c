import os
import time
import csv

from model.utils.test_policy import load_policy_and_env, run_policy
from model.synthetic import set_slo

NCONFS = [5, 10, 25, 50, 100]
NCOMPS = [3, 5, 10, 20]

EPISODE_COUNT = 20

for name, knob in [('std',0.05)]:
    slo = 150
    freq = int(1e6 / slo)
    print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")
    
    # Defaults to port 8000, which is fine for sequential execution
    set_slo(slo, freq, knob)

    time_tracker = []

    dir = 'data/'
    out_dir = 'results/raw/inference/infer_logs_seq/'
    
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    
    for nconf in NCONFS:
        for ncomp in NCOMPS:
            folder_name = f"{name}-f{nconf}-c{ncomp}"
            model_path = f"{dir}{folder_name}/{folder_name}_s0"
            
            # FIX: Create a unique output CSV file for every model
            out_path = os.path.join(out_dir, name, f'inference_{folder_name}.csv')
            
            params = {
                'log_dir': os.path.join(out_dir, name),
                'steps_per_epoch': 1000,
                'budget': [2500, 3600],
                'slo_latency': slo,
                'overrun_lim': 0.2,
                'mode': 'synthetic',
                'nconf': nconf,
                'ncomp': ncomp,
            }
            # FIX: Catch explicit exceptions so Ctrl+C works, and print the skips
            try:
                env, get_action = load_policy_and_env(model_path, 'last', params)
            except Exception as e:
                print(f"Skipping {folder_name}. Reason: {e}")
                continue

            start = time.time()
            run_policy(env, get_action, out_path, num_episodes=EPISODE_COUNT)
            end = time.time()

            time_tracker.append([name, nconf, ncomp, end - start])

    # FIX: Use write mode ('w') to prevent duplicate headers
    with open(f'results/raw/inference/infer_logs_seq/time_tracker_{name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Configs', 'Components', 'Time'])
        writer.writerows(time_tracker)
