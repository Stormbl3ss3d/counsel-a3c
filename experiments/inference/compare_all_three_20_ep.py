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

EPISODE_COUNT = 20
NUM_WORKERS = 4
EPISODES_PER_WORKER = EPISODE_COUNT // NUM_WORKERS  # 5 episodes per worker
OUTPUT_CSV = "results/inference/episode_20_inference_comparison.csv"


# =====================================================================
# 1. SEQUENTIAL: 1 Worker runs 20 episodes back-to-back
# =====================================================================
def run_sequential_20ep(model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    set_slo(slo, freq, knob, port=8000)
    env = CloudEnv('results/raw/inference/benchmark_logs/seq_20ep', 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=8000)
    
    t0 = time.time()
    total_steps = 0
    with torch.no_grad():
        for _ in range(EPISODE_COUNT):
            obs, mask = env.reset()
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                mask_t = torch.as_tensor(mask, dtype=torch.float32)
                action, _, _ = model.step(obs_t, mask_t)
                obs, mask, reward, done, _, _, _ = env.step(action)
                total_steps += 1
                
    elapsed = time.time() - t0
    env.close()
    return elapsed, total_steps


# =====================================================================
# 2. ASYNC PARALLEL: 4 Workers each run 5 episodes (20 total)
# =====================================================================
def async_ep_worker(worker_id, ready_barrier, start_barrier, step_counter, model, nconf, ncomp, slo, budget, overrun_lim, mode):
    port = 8000 + worker_id
    env = CloudEnv(f'results/raw/inference/benchmark_logs/par_20ep_w{worker_id}', 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)
    
    # Wait for all processes to finish spawning before starting timer
    ready_barrier.wait()
    start_barrier.wait()

    local_steps = 0
    with torch.no_grad():
        for _ in range(EPISODES_PER_WORKER):
            obs, mask = env.reset()
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                mask_t = torch.as_tensor(mask, dtype=torch.float32)
                action, _, _ = model.step(obs_t, mask_t)
                obs, mask, reward, done, _, _, _ = env.step(action)
                local_steps += 1

    with step_counter.get_lock():
        step_counter.value += local_steps
    env.close()


def run_async_parallel_20ep(model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    for w_id in range(NUM_WORKERS):
        set_slo(slo, freq, knob, port=8000 + w_id)

    ready_barrier = mp.Barrier(NUM_WORKERS + 1)
    start_barrier = mp.Barrier(NUM_WORKERS + 1)
    step_counter = mp.Value('i', 0)

    processes = []
    for w_id in range(NUM_WORKERS):
        p = mp.Process(target=async_ep_worker, args=(
            w_id, ready_barrier, start_barrier, step_counter, model, nconf, ncomp, slo, budget, overrun_lim, mode
        ))
        p.start()
        processes.append(p)

    ready_barrier.wait()
    t0 = time.time()
    start_barrier.wait()

    for p in processes:
        p.join()
    elapsed = time.time() - t0
    return elapsed, step_counter.value


# =====================================================================
# 3. BATCHED GPU: 4 Workers step in lockstep until 20 episodes finish
# =====================================================================
def pipe_ep_worker(remote, parent_remote, worker_id, nconf, ncomp, slo, budget, overrun_lim, mode):
    parent_remote.close()
    port = 8000 + worker_id
    env = CloudEnv(f'results/raw/inference/benchmark_logs/batch_20ep_w{worker_id}', 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, mask, reward, done, _, _, _ = env.step(data)
            if done:
                obs, mask = env.reset()
            remote.send((obs, mask, done))
        elif cmd == 'reset':
            obs, mask = env.reset()
            remote.send((obs, mask))
        elif cmd == 'close':
            env.close()
            remote.close()
            break


def run_batched_gpu_20ep(gpu_model, device, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode):
    for w_id in range(NUM_WORKERS):
        set_slo(slo, freq, knob, port=8000 + w_id)

    remotes, work_remotes = zip(*[mp.Pipe() for _ in range(NUM_WORKERS)])
    processes = []
    for worker_id, work_remote in enumerate(work_remotes):
        p = mp.Process(target=pipe_ep_worker, args=(
            work_remote, remotes[worker_id], worker_id, nconf, ncomp, slo, budget, overrun_lim, mode
        ))
        p.start()
        processes.append(p)
        work_remote.close()

    # Initial reset outside of loop
    t0 = time.time()
    for remote in remotes:
        remote.send(('reset', None))
    obs_list, mask_list = zip(*[remote.recv() for remote in remotes])

    eps_completed = 0
    total_steps = 0

    with torch.no_grad():
        # Stops as soon as 20 episodes total are completed across the 4 workers
        while eps_completed < EPISODE_COUNT:
            obs_batch = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)
            mask_batch = torch.as_tensor(np.array(mask_list), dtype=torch.float32, device=device)
            actions, _, _ = gpu_model.step(obs_batch, mask_batch)
            
            for i, remote in enumerate(remotes):
                remote.send(('step', actions[i]))
                
            obs_list, mask_list, dones = zip(*[remote.recv() for remote in remotes])
            eps_completed += sum(1 for d in dones if d)
            total_steps += NUM_WORKERS

    elapsed = time.time() - t0

    for remote in remotes:
        remote.send(('close', None))
    for p in processes:
        p.join()

    return elapsed, total_steps


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

    rows = []
    print(f"=== 20-Episode Benchmark Across All 3 Modes (Device: {device}) ===", flush=True)

    for nconf, ncomp in CONFIGS:
        folder_name = f"{name}-f{nconf}-c{ncomp}"
        model_file = f"a3c/{folder_name}/model/model1000.pt"
        if not os.path.exists(model_file):
            print(f"Skipping {folder_name}: not found.", flush=True)
            continue

        cpu_model = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
        cpu_model.eval()
        cpu_model.share_memory()

        gpu_model = torch.load(model_file, map_location=device, weights_only=False)
        gpu_model.eval()

        print(f"\nBenchmarking {folder_name} (20 Episodes)...", flush=True)
        seq_t, seq_steps = run_sequential_20ep(cpu_model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)
        par_t, par_steps = run_async_parallel_20ep(cpu_model, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)
        bat_t, bat_steps = run_batched_gpu_20ep(gpu_model, device, nconf, ncomp, slo, freq, knob, budget, overrun_lim, mode)

        print(f"  Sequential   : {seq_t:6.2f}s ({seq_steps} steps)", flush=True)
        print(f"  Async-Par    : {par_t:6.2f}s ({par_steps} steps) | Speedup: {seq_t/par_t:.2f}x", flush=True)
        print(f"  Batched-GPU  : {bat_t:6.2f}s ({bat_steps} steps) | Speedup: {seq_t/bat_t:.2f}x", flush=True)

        rows.append({
            'Model': folder_name,
            'Configs': nconf,
            'Components': ncomp,
            'Seq_20Ep_Time_s': round(seq_t, 3),
            'AsyncPar_20Ep_Time_s': round(par_t, 3),
            'BatchedGPU_20Ep_Time_s': round(bat_t, 3),
            'AsyncPar_Speedup': round(seq_t / par_t, 2),
            'BatchedGPU_Speedup': round(seq_t / bat_t, 2),
            'Seq_Total_Steps': seq_steps,
            'AsyncPar_Total_Steps': par_steps,
            'BatchedGPU_Total_Steps': bat_steps
        })

    if rows:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved 20-episode comparison to {OUTPUT_CSV}", flush=True)
