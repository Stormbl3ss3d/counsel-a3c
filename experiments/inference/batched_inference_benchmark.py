import os
import csv
import time
import torch
import torch.multiprocessing as mp
import numpy as np
from model.env import CloudEnv
from model.synthetic import set_slo

# The 8 standard COUNSEL (nconf, ncomp) model combinations
# (2 configuration sizes x 4 component chain lengths)
MODEL_CONFIGS = [
    (5, 3), (5, 5), (5, 10), (5, 20),
    (10, 3), (25, 3), (50, 3), (100, 3)
]

NUM_WORKERS = 4
EPISODES_PER_WORKER = 5  # 4 workers * 5 episodes = 20 episodes total per model
TOTAL_EPISODES = NUM_WORKERS * EPISODES_PER_WORKER
OUTPUT_CSV = "results/inference/batched_inference_times.csv"


def environment_worker(remote, parent_remote, worker_id, nconf, ncomp, slo, budget, overrun_lim, mode):
    parent_remote.close()
    port = 8000 + worker_id
    log_dir = f'results/raw/inference/benchmark_logs/batched_f{nconf}_c{ncomp}_worker_{worker_id}'
    os.makedirs(log_dir, exist_ok=True)
    
    env = CloudEnv(log_dir, 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)
    
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, mask, reward, done, _, _, _ = env.step(data)
            if done:
                obs, mask = env.reset()
            # Return 'done' flag as well so the main process can count completed episodes
            remote.send((obs, mask, done))
        elif cmd == 'reset':
            obs, mask = env.reset()
            remote.send((obs, mask))
        elif cmd == 'close':
            env.close()
            remote.close()
            break


def benchmark_single_model(nconf, ncomp, device):
    folder_name = f"std-f{nconf}-c{ncomp}"
    model_file = f"a3c/{folder_name}/model/model1000.pt"
    
    if not os.path.exists(model_file):
        print(f"[SKIP] Model file not found: {model_file}", flush=True)
        return None

    slo = 150
    freq = int(1e6 / slo)
    knob = 0.05
    budget = [2500, 3600]
    overrun_lim = 0.2
    mode = 'synthetic'

    # Reset SLO on all 4 loadservers before starting this model's benchmark
    for w_id in range(NUM_WORKERS):
        set_slo(slo, freq, knob, port=8000 + w_id)

    remotes, work_remotes = zip(*[mp.Pipe() for _ in range(NUM_WORKERS)])
    processes = []
    
    for worker_id, work_remote in enumerate(work_remotes):
        p = mp.Process(target=environment_worker, args=(
            work_remote, remotes[worker_id], worker_id, nconf, ncomp, slo, budget, overrun_lim, mode
        ))
        p.start()
        processes.append(p)
        work_remote.close()

    model = torch.load(model_file, map_location=device, weights_only=False)
    model.eval()

    # Initial parallel reset
    for remote in remotes:
        remote.send(('reset', None))
    results = [remote.recv() for remote in remotes]
    obs_list, mask_list = zip(*results)

    episodes_completed = [0] * NUM_WORKERS
    total_steps = 0
    batch_passes = 0

    print(f"\n---> Benchmarking {folder_name} ({TOTAL_EPISODES} total episodes across {NUM_WORKERS} workers)...", flush=True)
    start_time = time.time()

    with torch.no_grad():
        # Continue synchronous batching until every worker finishes at least EPISODES_PER_WORKER (5) episodes
        while any(ep < EPISODES_PER_WORKER for ep in episodes_completed):
            obs_batch = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)
            mask_batch = torch.as_tensor(np.array(mask_list), dtype=torch.float32, device=device)
            
            actions, _, _ = model.step(obs_batch, mask_batch)
            actions_cpu = actions
            
            for i, remote in enumerate(remotes):
                remote.send(('step', actions_cpu[i]))
            
            step_results = [remote.recv() for remote in remotes]
            obs_list, mask_list, dones = zip(*step_results)
            
            for i, is_done in enumerate(dones):
                if is_done:
                    episodes_completed[i] += 1
            
            total_steps += NUM_WORKERS
            batch_passes += 1

    elapsed_time = time.time() - start_time

    # Cleanly shut down the worker processes for this (nconf, ncomp) configuration
    for remote in remotes:
        remote.send(('close', None))
    for p in processes:
        p.join()

    total_eps_run = sum(episodes_completed)
    time_per_ep = elapsed_time / TOTAL_EPISODES
    steps_per_sec = total_steps / elapsed_time

    print(
        f"[DONE] {folder_name} | Time: {elapsed_time:.2f}s | "
        f"Steps: {total_steps} ({batch_passes} batches) | "
        f"Throughput: {steps_per_sec:.1f} steps/s",
        flush=True
    )

    return {
        "model": folder_name,
        "nconf": nconf,
        "ncomp": ncomp,
        "num_workers": NUM_WORKERS,
        "target_episodes": TOTAL_EPISODES,
        "actual_episodes_completed": total_eps_run,
        "total_steps": total_steps,
        "batch_forward_passes": batch_passes,
        "batched_time_sec": round(elapsed_time, 4),
        "avg_time_per_episode_sec": round(time_per_ep, 4),
        "throughput_steps_per_sec": round(steps_per_sec, 2)
    }


def run_all_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing 8-model batched inference benchmark on: {device}", flush=True)

    summary_rows = []

    for nconf, ncomp in MODEL_CONFIGS:
        metrics = benchmark_single_model(nconf, ncomp, device)
        if metrics is not None:
            summary_rows.append(metrics)

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        
        print(f"\n=======================================================", flush=True)
        print(f"Benchmark complete! Saved results to: {OUTPUT_CSV}", flush=True)
        print(f"=======================================================", flush=True)
        
        # Print a clean summary table to the terminal
        print(f"{'Model':<12} | {'nconf':<5} | {'ncomp':<5} | {'Batched Time (s)':<16} | {'Steps/sec':<10}", flush=True)
        print("-" * 60, flush=True)
        for row in summary_rows:
            print(
                f"{row['model']:<12} | {row['nconf']:<5} | {row['ncomp']:<5} | "
                f"{row['batched_time_sec']:<16.2f} | {row['throughput_steps_per_sec']:<10.1f}",
                flush=True
            )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    run_all_benchmarks()
