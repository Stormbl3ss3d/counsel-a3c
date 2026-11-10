import os
import csv
import time
import torch
import torch.multiprocessing as mp
import numpy as np

from model.env import CloudEnv
from model.synthetic import set_slo

CONFIGS = [
    (5, 3), (5, 5), (5, 10), (5, 20),
    (10, 3), (25, 3), (50, 3), (100, 3)
]

# The benchmark constraint: exactly 2000 environment steps total.
# Sequential runs 2000 steps. Parallel runs 500 steps per worker (x4 workers).
TOTAL_STEPS = 2000  
NUM_WORKERS = 4


def run_worker_steps(worker_id, model, folder_name, nconf, ncomp, slo, steps_to_run, budget, overrun_lim, mode):
    """Isolated function for parallel workers to execute an exact number of steps."""
    port = 8000 + worker_id
    log_dir = f'results/raw/inference/benchmark_logs/worker_{worker_id}'
    
    env = CloudEnv(log_dir, 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)
    obs, mask = env.reset()
    
    with torch.no_grad():
        for _ in range(steps_to_run):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mask_t = torch.as_tensor(mask, dtype=torch.float32)
            
            action, _, _ = model.step(obs_t, mask_t)
            obs, mask, reward, done, _, _, _ = env.step(action)
            
            # If an episode finishes early, immediately reset and keep counting steps
            if done:
                obs, mask = env.reset()
                
    env.close()


def benchmark_sequential(model, folder_name, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    """Benchmarks a single worker running the full step count sequentially."""
    port = 8000
    set_slo(slo, freq, knob, port=port)
    
    log_dir = 'results/raw/inference/benchmark_logs/seq'
    env = CloudEnv(log_dir, 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)
    obs, mask = env.reset()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(TOTAL_STEPS):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mask_t = torch.as_tensor(mask, dtype=torch.float32)
            
            action, _, _ = model.step(obs_t, mask_t)
            obs, mask, reward, done, _, _, _ = env.step(action)
            
            if done:
                obs, mask = env.reset()
                
    end_time = time.time()
    env.close()
    
    return end_time - start_time


def benchmark_parallel(model, folder_name, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    """Benchmarks multiple workers splitting the step count simultaneously."""
    steps_per_worker = TOTAL_STEPS // NUM_WORKERS
    
    # Initialize all loadservers
    for w_id in range(NUM_WORKERS):
        set_slo(slo, freq, knob, port=8000 + w_id)
        
    processes = []
    
    start_time = time.time()
    for w_id in range(NUM_WORKERS):
        p = mp.Process(target=run_worker_steps, args=(
            w_id, model, folder_name, nconf, ncomp, slo, 
            steps_per_worker, budget, overrun_lim, mode
        ))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    end_time = time.time()
    
    return end_time - start_time


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    name = 'std'
    knob = 0.05
    slo = 150
    freq = int(1e6 / slo)
    budget = [2500, 3600]
    overrun_lim = 0.2
    mode = 'synthetic'

    os.makedirs('results/raw/inference/benchmark_logs', exist_ok=True)
    out_csv = 'results/inference/benchmark_throughput_results.csv'

    print(f"Starting Throughput Benchmark (Target: {TOTAL_STEPS} Steps) \n", flush=True)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Configs', 'Components', 'Seq_Time(s)', 'Par_Time(s)', 'Speedup'])

        for nconf, ncomp in CONFIGS:
            folder_name = f"{name}-f{nconf}-c{ncomp}"
            model_file = f"a3c/{folder_name}/model/model1000.pt"
            
            if not os.path.exists(model_file):
                print(f"Skipping {folder_name}: {model_file} not found.", flush=True)
                continue
                
            print(f"Benchmarking {folder_name}...", flush=True)
            
            # Load model once and share memory for parallel compatibility
            model = torch.load(model_file, map_location=torch.device('cpu'))
            model.eval()
            model.share_memory()
            
            # 1. Run Sequential Benchmark
            seq_time = benchmark_sequential(model, folder_name, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)
            
            # 2. Run Parallel Benchmark
            par_time = benchmark_parallel(model, folder_name, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)
            
            speedup = seq_time / par_time if par_time > 0 else 0
            
            print(f"Sequential Time: {seq_time:.2f}s", flush=True)
            print(f"Parallel Time:   {par_time:.2f}s", flush=True)
            print(f"True Speedup:    {speedup:.2f}x\n", flush=True)
            
            writer.writerow([nconf, ncomp, seq_time, par_time, speedup])

    print(f"Benchmark complete! Results saved to {out_csv}", flush=True)