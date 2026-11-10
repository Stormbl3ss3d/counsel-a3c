import time
import torch
import torch.multiprocessing as mp
import numpy as np
from model.env import CloudEnv
from model.synthetic import set_slo

TOTAL_STEPS = 2000
NUM_WORKERS = 4
TARGET_CONFIG = (5, 10)

def environment_worker(remote, parent_remote, worker_id, nconf, ncomp, slo, budget, overrun_lim, mode):
    parent_remote.close()
    port = 8000 + worker_id
    log_dir = f'results/raw/inference/benchmark_logs/batched_worker_{worker_id}'
    env = CloudEnv(log_dir, 1000, budget, slo, overrun_lim, mode, nconf, ncomp, port=port)
    
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

def run_batched_inference():
    # Target your NVIDIA 4050
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing batched inference on: {device}", flush=True)
    if device.type != 'cuda':
        print("WARNING: CUDA not detected in container. Falling back to CPU.", flush=True)

    nconf, ncomp = TARGET_CONFIG
    folder_name = f"std-f{nconf}-c{ncomp}"
    model_file = f"a3c/{folder_name}/model/model1000.pt"
    
    slo = 150
    freq = int(1e6 / slo)
    knob = 0.05
    budget = [2500, 3600]
    overrun_lim = 0.2
    mode = 'synthetic'

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

    for remote in remotes:
        remote.send(('reset', None))
    
    results = [remote.recv() for remote in remotes]
    obs_list, mask_list = zip(*results)

    print("Starting synchronous batched steps...", flush=True)
    start_time = time.time()
    steps_taken = 0

    with torch.no_grad():
        while steps_taken < TOTAL_STEPS:
            obs_batch = torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device)
            mask_batch = torch.as_tensor(np.array(mask_list), dtype=torch.float32, device=device)
            
            actions, _, _ = model.step(obs_batch, mask_batch)
            actions_cpu = actions
            
            for i, remote in enumerate(remotes):
                remote.send(('step', actions_cpu[i]))
            
            results = [remote.recv() for remote in remotes]
            obs_list, mask_list = zip(*results)
            
            steps_taken += NUM_WORKERS

    end_time = time.time()
    
    for remote in remotes:
        remote.send(('close', None))
    for p in processes:
        p.join()
        
    print(f"Completed {TOTAL_STEPS} batched steps in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    run_batched_inference()
