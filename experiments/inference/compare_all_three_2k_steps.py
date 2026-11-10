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

TOTAL_STEPS = 2000  # Fixed step limit for apples-to-apples throughput comparison
NUM_WORKERS = 4
OUTPUT_CSV = "results/inference/master_inference_comparison.csv"


# =====================================================================
# 1. SEQUENTIAL (1 Worker)
# =====================================================================
def run_sequential(model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    set_slo(slo, freq, knob, port=8000)
    env = CloudEnv('results/raw/inference/benchmark_logs/seq', 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=8000)
    
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
                
    elapsed = time.time() - start_time
    env.close()
    return elapsed


# =====================================================================
# 2. ASYNC PARALLEL (4 Workers, Shared CPU Model)
# =====================================================================
def async_worker(worker_id, ready_barrier, start_barrier, model, nconf, ncomp, slo, budget, overrun_lim, mode, steps_per_worker):
    port = 8000 + worker_id
    env = CloudEnv(f'results/raw/inference/benchmark_logs/par_w{worker_id}', 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)
    obs, mask = env.reset()
    
    ready_barrier.wait()
    start_barrier.wait()

    with torch.no_grad():
        for _ in range(steps_per_worker):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mask_t = torch.as_tensor(mask, dtype=torch.float32)
            action, _, _ = model.step(obs_t, mask_t)
            obs, mask, reward, done, _, _, _ = env.step(action)
            if done:
                obs, mask = env.reset()
                
    env.close()


def run_async_parallel(model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    steps_per_worker = TOTAL_STEPS // NUM_WORKERS
    for w_id in range(NUM_WORKERS):
        set_slo(slo, freq, knob, port=8000 + w_id)

    ready_barrier = mp.Barrier(NUM_WORKERS + 1)
    start_barrier = mp.Barrier(NUM_WORKERS + 1)

    processes = []
    for w_id in range(NUM_WORKERS):
        p = mp.Process(target=async_worker, args=(
            w_id, ready_barrier, start_barrier, model, nconf, ncomp, slo, budget, overrun_lim, mode, steps_per_worker
        ))
        p.start()
        processes.append(p)

    ready_barrier.wait()
    start_time = time.time()
    start_barrier.wait()

    for p in processes:
        p.join()
        
    return time.time() - start_time


# =====================================================================
# 3. BATCHED GPU (4 Workers via IPC Pipe + Stacked CUDA Tensor)
# =====================================================================
def pipe_worker(remote, parent_remote, worker_id, nconf, ncomp, slo, budget, overrun_lim, mode):
    parent_remote.close()
    port = 8000 + worker_id
    env = CloudEnv(f'results/raw/inference/benchmark_logs/batch_w{worker_id}', 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, mask, reward, done, _, _, _ = env.step(data)
            if done:
                obs, mask = env.reset()
            remote.send((obs, mask))
        elif cmd == 'reset':
            obs, mask = env.reset()
            remote.send((obs, mask))
        elif cmd == 'close':
            env.close()
            remote.close()
            break


def run_batched_gpu(gpu_model, device, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    for w_id in range(NUM_WORKERS):
        set_slo(slo, freq, knob, port=8000 + w_id)

    remotes, work_remotes = zip(*[mp.Pipe() for _ in range(NUM_WORKERS)])
    processes = []
    for worker_id, work_remote in enumerate(work_remotes):
        p = mp.Process(target=pipe_worker, args=(
            work_remote, remotes[worker_id], worker_id, nconf, ncomp, slo, budget, overrun_lim, mode
        ))
        p.start()
        processes.append(p)
        work_remote.close()

    for remote in remotes:
        remote.send(('reset', None))
    obs_list, mask_list = zip(*[remote.recv() for remote in remotes])

    start_time = time.time()
    steps_taken = 0

    with torch.no_grad():
        while steps_taken < TOTAL_STEPS:
            obs_batch = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)
            mask_batch = torch.as_tensor(np.array(mask_list), dtype=torch.float32, device=device)
            
            actions, _, _ = gpu_model.step(obs_batch, mask_batch)
            
            for i, remote in enumerate(remotes):
                remote.send(('step', actions[i]))
                
            obs_list, mask_list = zip(*[remote.recv() for remote in remotes])
            steps_taken += NUM_WORKERS

    elapsed = time.time() - start_time

    for remote in remotes:
        remote.send(('close', None))
    for p in processes:
        p.join()

    return elapsed


# =====================================================================
# Main Benchmark Runner
# =====================================================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    name = 'std'
    knob = 0.05
    slo = 150
    freq = int(1e6 / slo)
    budget = [2500, 3600]
    overrun_lim = 0.2
    mode = 'synthetic'

    os.makedirs('results/raw/inference/benchmark_logs', exist_ok=True)
    summary_rows = []

    print(f"=== Starting 3-Way Inference Comparison ({TOTAL_STEPS} Steps) on: {device} ===\n", flush=True)

    for nconf, ncomp in CONFIGS:
        folder_name = f"{name}-f{nconf}-c{ncomp}"
        model_file = f"a3c/{folder_name}/model/model1000.pt"
        if not os.path.exists(model_file):
            print(f"Skipping {folder_name}: model not found.", flush=True)
            continue

        print(f"Benchmarking {folder_name}...", flush=True)

        # 1. Load CPU model for Sequential & Async-Parallel
        cpu_model = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
        cpu_model.eval()
        cpu_model.share_memory()

        # 2. Load GPU model for Batched-GPU
        gpu_model = torch.load(model_file, map_location=device, weights_only=False)
        gpu_model.eval()

        # Execute all three modes
        seq_time = run_sequential(cpu_model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)
        par_time = run_async_parallel(cpu_model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)
        bat_time = run_batched_gpu(gpu_model, device, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)

        par_speedup = seq_time / par_time if par_time > 0 else 0
        bat_speedup = seq_time / bat_time if bat_time > 0 else 0

        print(f"  -> Sequential:   {seq_time:6.2f}s ({TOTAL_STEPS/seq_time:5.1f} steps/s)")
        print(f"  -> Async-Par:    {par_time:6.2f}s ({TOTAL_STEPS/par_time:5.1f} steps/s) | Speedup: {par_speedup:.2f}x")
        print(f"  -> Batched-GPU:  {bat_time:6.2f}s ({TOTAL_STEPS/bat_time:5.1f} steps/s) | Speedup: {bat_speedup:.2f}x\n")

        summary_rows.append({
            'Configs': nconf,
            'Components': ncomp,
            'Seq_Time_s': round(seq_time, 3),
            'AsyncPar_Time_s': round(par_time, 3),
            'BatchedGPU_Time_s': round(bat_time, 3),
            'AsyncPar_Speedup': round(par_speedup, 2),
            'BatchedGPU_Speedup': round(bat_speedup, 2)
        })

    if summary_rows:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"=======================================================================", flush=True)
        print(f"Comparison complete! Saved results to: {OUTPUT_CSV}", flush=True)
        print(f"=======================================================================", flush=True)
        
        # Print formatted table
        print(f"{'Config':<8} | {'Seq (s)':<8} | {'Par (s)':<8} | {'Batched (s)':<11} | {'Par Speedup':<12} | {'Batched Speedup'}", flush=True)
        print("-" * 72, flush=True)
        for row in summary_rows:
            cfg_str = f"f{row['Configs']}-c{row['Components']}"
            print(
                f"{cfg_str:<8} | {row['Seq_Time_s']:<8.2f} | {row['AsyncPar_Time_s']:<8.2f} | "
                f"{row['BatchedGPU_Time_s']:<11.2f} | {row['AsyncPar_Speedup']:<12.2f}x | {row['BatchedGPU_Speedup']:.2f}x",
                flush=True
            )
