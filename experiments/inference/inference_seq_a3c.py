import os
import time
import csv
import torch

from model.env import CloudEnv
from model.synthetic import set_slo

CONFIGS = [
    (5, 3), (5, 5), (5, 10), (5, 20),
    (10, 3), (25, 3), (50, 3), (100, 3)
]

EPISODE_COUNT = 20

if __name__ == '__main__':
    name = 'std'
    knob = 0.05
    slo = 150
    freq = int(1e6 / slo)

    print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")
    # Initialize a single loadserver for sequential execution
    set_slo(slo, freq, knob, port=8000)

    time_tracker = []
    out_dir = 'results/raw/inference/infer_logs_seq/'
    os.makedirs(out_dir, exist_ok=True)

    for nconf, ncomp in CONFIGS:
        folder_name = f"{name}-f{nconf}-c{ncomp}"
        model_file = f"a3c/{folder_name}/model/model1000.pt"
        
        if not os.path.exists(model_file):
            print(f"Skipping {folder_name}: {model_file} not found.")
            continue

        # Load the exact same A3C model object used in parallel inference
        model = torch.load(model_file, map_location=torch.device('cpu'))
        model.eval()

        path = os.path.join(out_dir, folder_name)
        steps_per_epoch = 1000
        budget = [2500, 3600]
        overrun_lim = 0.2
        mode = 'synthetic'

        # Single environment pointing to loadserver_0
        env = CloudEnv(
            path, steps_per_epoch, budget, slo,
            overrun_lim, mode, nconf, ncomp, port=8000
        )

        start = time.time()
        
        with torch.no_grad():
            for ep in range(EPISODE_COUNT):
                obs, mask = env.reset()
                done = False
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    mask_t = torch.as_tensor(mask, dtype=torch.float32)
                    action, _, _ = model.step(obs_t, mask_t)
                    obs, mask, reward, done, _, _, _ = env.step(action)
                    
        end = time.time()
        env.close()

        total_time = end - start
        print(f"Completed {folder_name} sequentially in {total_time:.2f}s")
        time_tracker.append([name, nconf, ncomp, total_time])

    out_csv = os.path.join(out_dir, f'time_tracker_{name}_true_baseline.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Configs', 'Components', 'Time'])
        writer.writerows(time_tracker)

    print(f"\nTrue sequential baseline saved to: {out_csv}")