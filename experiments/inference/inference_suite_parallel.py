import os
import time
import csv
import queue
import torch
import torch.multiprocessing as mp

from model.env import CloudEnv
from model.synthetic import set_slo

CONFIGS = [
    (5, 3), (5, 5), (5, 10), (5, 20),
    (10, 3), (25, 3), (50, 3), (100, 3)
]

EPISODE_COUNT = 20
NUM_WORKERS = 4  # Matches your 4 loadserver containers

def model_worker(worker_id, task_queue, result_queue, name, slo, freq, knob, budget, overrun_lim, mode, steps_per_epoch):
    """Worker process that consumes model configurations from a queue and runs 20 episodes."""
    port = 8000 + worker_id

    while True:
        try:
            # Safely pop the next model configuration off the queue
            nconf, ncomp = task_queue.get_nowait()
        except queue.Empty:
            # Queue is empty, all models have been processed
            break

        folder_name = f"{name}-f{nconf}-c{ncomp}"
        model_file = f"a3c/{folder_name}/model/model1000.pt"

        if not os.path.exists(model_file):
            print(f"[Worker {worker_id}] Skipping {folder_name}: Not found.", flush=True)
            continue

        # Initialize the SLO on this worker's dedicated loadserver
        set_slo(slo, freq, knob, port=port)
        
        # Load this specific model configuration into the worker's local memory
        model = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
        model.eval()

        log_dir = f'results/raw/inference/infer_logs_suite/{folder_name}'
        env = CloudEnv(log_dir, steps_per_epoch, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)

        print(f"[Worker {worker_id}] Starting {EPISODE_COUNT} episodes for {folder_name}...", flush=True)
        start_time = time.time()

        with torch.no_grad():
            for ep in range(EPISODE_COUNT):
                obs, mask = env.reset()
                done = False
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    mask_t = torch.as_tensor(mask, dtype=torch.float32)
                    action, _, _ = model.step(obs_t, mask_t)
                    obs, mask, reward, done, _, _, _ = env.step(action)

        elapsed = time.time() - start_time
        env.close()
        
        print(f"[Worker {worker_id}] Completed {folder_name} in {elapsed:.2f}s", flush=True)
        result_queue.put([name, nconf, ncomp, elapsed])


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    name = 'std'
    knob = 0.05
    slo = 150
    freq = int(1e6 / slo)
    budget = [2500, 3600]
    overrun_lim = 0.2
    mode = 'synthetic'
    steps_per_epoch = 1000
    
    out_dir = 'results/raw/inference/infer_logs_suite/'
    os.makedirs(out_dir, exist_ok=True)

    # Use multiprocessing queues to distribute the 8 models across the 4 workers
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for conf in CONFIGS:
        task_queue.put(conf)

    processes = []
    wall_start = time.time()

    print(f"=== Starting Parallel Suite Evaluation ({len(CONFIGS)} Models across {NUM_WORKERS} Workers) ===", flush=True)

    for w_id in range(NUM_WORKERS):
        p = mp.Process(
            target=model_worker,
            args=(w_id, task_queue, result_queue, name, slo, freq, knob, budget, overrun_lim, mode, steps_per_epoch)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_wall_time = time.time() - wall_start
    print(f"\n=======================================================", flush=True)
    print(f"Suite evaluation complete in {total_wall_time:.2f} total seconds.")
    print(f"=======================================================", flush=True)

    # Collect and save results to CSV
    time_tracker = []
    while not result_queue.empty():
        time_tracker.append(result_queue.get())

    out_csv = os.path.join(out_dir, f"time_tracker_{name}_suite.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Configs', 'Components', 'Time'])
        writer.writerows(time_tracker)

    print(f"Model parallel inference benchmark saved to: {out_csv}", flush=True)
