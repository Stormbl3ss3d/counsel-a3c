import os
import time
import csv
import torch
import torch.multiprocessing as mp

from model.env import CloudEnv
from model.synthetic import set_slo

CONFIGS = [
    (5, 3), (5, 5), (5, 10), (5, 20),
    (10, 3), (25, 3), (50, 3), (100, 3)
]

EPISODE_COUNT = 20
NUM_WORKERS = 4
EPISODES_PER_WORKER = EPISODE_COUNT // NUM_WORKERS  # 5 episodes per worker


def run_worker_episodes(worker_id, model, cloud_conf, num_episodes, return_dict):
    """Worker process: runs evaluation episodes against its dedicated loadserver."""
    port = 8000 + worker_id
    log_dir = f"{cloud_conf[0]}/w{worker_id:02d}"
    
    # Initialize the worker's dedicated environment pointing to its specific port
    env = CloudEnv(
        log_dir, cloud_conf[1], cloud_conf[2], cloud_conf[3],
        cloud_conf[4], cloud_conf[5], cloud_conf[6], cloud_conf[7],
        port=port
    )

    all_rewards = []
    start_time = time.time()

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, mask = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                mask_t = torch.as_tensor(mask, dtype=torch.float32)
                action, _, _ = model.step(obs_t, mask_t)

                obs, mask, reward, done, _, _, _ = env.step(action)
                ep_reward += reward

            all_rewards.append(ep_reward)

    env.close()
    elapsed = time.time() - start_time
    return_dict[worker_id] = (elapsed, all_rewards)


def evaluate_parallel(name, nconf, ncomp, slo, knob, freq, out_dir):
    model_dir = f"a3c/{name}-f{nconf}-c{ncomp}/model"
    model_file = os.path.join(model_dir, "model1000.pt")

    if not os.path.exists(model_file):
        print(f"Skipping {name}-f{nconf}-c{ncomp}: {model_file} not found.")
        return None

    # Load the full GCNActorCritic model object
    model = torch.load(model_file, map_location=torch.device('cpu'))
    model.eval()
    model.share_memory()

    # Configure loadservers
    for w_id in range(NUM_WORKERS):
        set_slo(slo, freq, knob, port=8000 + w_id)

    path = os.path.join(out_dir, f"{name}-f{nconf}-c{ncomp}")
    steps_per_epoch = 1000
    budget = [2500, 3600]
    overrun_lim = 0.2
    mode = 'synthetic'

    cloud_conf = [path, steps_per_epoch, budget, slo, overrun_lim, mode, nconf, ncomp]

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    wall_start = time.time()

    for w_id in range(NUM_WORKERS):
        p = mp.Process(
            target=run_worker_episodes,
            args=(w_id, model, cloud_conf, EPISODES_PER_WORKER, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_wall_time = time.time() - wall_start
    print(f"Completed {name}-f{nconf}-c{ncomp} in {total_wall_time:.2f}s")
    return [name, nconf, ncomp, total_wall_time]


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    name = 'std'
    knob = 0.05
    slo = 150
    freq = int(1e6 / slo)
    out_dir = 'results/raw/inference/infer_logs_parallel/'

    os.makedirs(out_dir, exist_ok=True)
    time_tracker = []

    for nconf, ncomp in CONFIGS:
        res = evaluate_parallel(name, nconf, ncomp, slo, knob, freq, out_dir)
        if res:
            time_tracker.append(res)

    out_csv = os.path.join(out_dir, f"time_tracker_{name}.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Configs', 'Components', 'Time'])
        writer.writerows(time_tracker)

    print(f"\nParallel inference benchmark saved to: {out_csv}")