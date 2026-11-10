import os
import csv
import time
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd

from model.env import CloudEnv
from model.synthetic import set_slo

TOTAL_STEPS = 2000  
WORKER_COUNTS = [1, 2, 4, 8]
TARGET_CONFIG = (5, 10) # Using 5-10 as it showed the best scaling previously

def run_worker_steps(worker_id, model, nconf, ncomp, slo, steps_to_run, budget, overrun_lim, mode):
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
            if done:
                obs, mask = env.reset()
    env.close()

def benchmark_workers(num_workers, model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    steps_per_worker = TOTAL_STEPS // num_workers
    for w_id in range(num_workers):
        set_slo(slo, freq, knob, port=8000 + w_id)
        
    processes = []
    start_time = time.time()
    for w_id in range(num_workers):
        p = mp.Process(target=run_worker_steps, args=(
            w_id, model, nconf, ncomp, slo, steps_per_worker, budget, overrun_lim, mode
        ))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    return time.time() - start_time

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    nconf, ncomp = TARGET_CONFIG
    folder_name = f"std-f{nconf}-c{ncomp}"
    model_file = f"a3c/{folder_name}/model/model1000.pt"
    
    slo = 150
    freq = int(1e6 / slo)
    knob = 0.05
    budget = [2500, 3600]
    overrun_lim = 0.2
    mode = 'synthetic'

    out_csv = 'results/inference/scaling_results.csv'
    results = []

    print(f"=== Starting Scaling Demo for Config {nconf}-{ncomp} ===", flush=True)
    
    model = torch.load(model_file, map_location=torch.device('cpu'))
    model.eval()
    model.share_memory()

    # 1. Run Benchmarks
    baseline_time = None
    for w in WORKER_COUNTS:
        print(f"\nEvaluating with {w} worker(s)...", flush=True)
        t = benchmark_workers(w, model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)
        
        if w == 1:
            baseline_time = t
            
        speedup = baseline_time / t
        results.append({"Workers": w, "Time (s)": t, "Speedup": speedup})
        print(f" -> Time: {t:.2f}s | Speedup: {speedup:.2f}x", flush=True)

    # 2. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}", flush=True)

    # 3. Generate Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Number of Workers')
    ax1.set_ylabel('Execution Time (seconds)', color=color)
    ax1.plot(df['Workers'], df['Time (s)'], marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(WORKER_COUNTS)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Speedup Factor', color=color)  
    ax2.plot(df['Workers'], df['Speedup'], marker='s', linestyle='--', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'A3C Inference Scaling Performance ({TOTAL_STEPS} Steps, Config {nconf}-{ncomp})')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plot_path = 'results/figures/scaling_plot.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}", flush=True)
