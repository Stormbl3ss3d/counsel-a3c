import os
import csv
import torch
import numpy as np

from model.env import CloudEnv
from model.synthetic import set_slo
from model.utils.test_policy import load_policy_and_env

CONFIGS = [
    (5, 3), (5, 5), (5, 10), (5, 20),
    (10, 3), (25, 3), (50, 3), (100, 3)
]

EPISODE_COUNT = 20

def evaluate_baseline(base_model_path, params, out_csv):
    """Evaluates the standard model using a manual loop to capture custom metrics."""
    if not os.path.exists(base_model_path):
        print(f"Skipping Baseline: {base_model_path} not found.", flush=True)
        return None, None, None

    env_base, get_action_base = load_policy_and_env(base_model_path, 'last', params)
    
    ep_rewards = []
    all_utils = []
    total_breaches = 0

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Step', 'Reward', 'Utilization', 'Breach'])

        for ep in range(EPISODE_COUNT):
            obs, mask = env_base.reset()
            done = False
            steps = 0
            ep_reward = 0.0

            while not done and steps < params['steps_per_epoch']:
                # SpinningUp's loaded policy function handles its own tensor conversions
                action = get_action_base(obs,mask)
                obs, mask, reward, done, _, end_util, breach = env_base.step(action)

                ep_reward += reward
                util_val = float(np.mean(end_util)) if isinstance(end_util, (list, tuple, np.ndarray)) else float(end_util)
                
                all_utils.append(util_val)
                if breach:
                    total_breaches += 1

                writer.writerow([ep + 1, steps + 1, reward, util_val, int(bool(breach))])
                steps += 1

            ep_rewards.append(ep_reward)

    env_base.close()
    avg_reward = np.mean(ep_rewards)
    avg_util = np.mean(all_utils)
    return avg_reward, avg_util, total_breaches


def evaluate_a3c(folder_name, nconf, ncomp, slo, steps_per_epoch, budget, overrun_lim, mode, out_csv):
    """Evaluates the A3C PyTorch model and logs step-by-step metrics."""
    model_file = f"a3c/{folder_name}/model/model1000.pt"
    if not os.path.exists(model_file):
        print(f"Skipping A3C {folder_name}: {model_file} not found.", flush=True)
        return None, None, None

    model = torch.load(model_file, map_location=torch.device('cpu'))
    model.eval()

    log_path = os.path.dirname(out_csv)
    env = CloudEnv(log_path, steps_per_epoch, budget, slo, overrun_lim, mode, nconf, ncomp, port=8000)

    ep_rewards = []
    all_utils = []
    total_breaches = 0

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Step', 'Reward', 'Utilization', 'Breach'])

        with torch.no_grad():
            for ep in range(EPISODE_COUNT):
                obs, mask = env.reset()
                done = False
                steps = 0
                ep_reward = 0.0

                while not done and steps < steps_per_epoch:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    mask_t = torch.as_tensor(mask, dtype=torch.float32)

                    action, _, _ = model.step(obs_t, mask_t)
                    obs, mask, reward, done, _, end_util, breach = env.step(action)

                    ep_reward += reward
                    util_val = float(np.mean(end_util)) if isinstance(end_util, (list, tuple, np.ndarray)) else float(end_util)
                    
                    all_utils.append(util_val)
                    if breach:
                        total_breaches += 1

                    writer.writerow([ep + 1, steps + 1, reward, util_val, int(bool(breach))])
                    steps += 1

                ep_rewards.append(ep_reward)

    env.close()
    avg_reward = np.mean(ep_rewards)
    avg_util = np.mean(all_utils)
    return avg_reward, avg_util, total_breaches


if __name__ == '__main__':
    name = 'std'
    knob = 0.05
    slo = 150
    freq = int(1e6 / slo)

    steps_per_epoch = 1000
    budget = [2500, 3600]
    overrun_lim = 0.2
    mode = 'synthetic'

    root_out = 'results/raw/inference/quality_eval_logs'
    baseline_out_dir = os.path.join(root_out, 'baseline')
    a3c_out_dir = os.path.join(root_out, 'a3c')

    os.makedirs(baseline_out_dir, exist_ok=True)
    os.makedirs(a3c_out_dir, exist_ok=True)

    summary_records = []

    print(f"=== Starting Model Quality Comparison (SLO: {slo}, Knob: {knob}) ===", flush=True)

    for nconf, ncomp in CONFIGS:
        folder_name = f"{name}-f{nconf}-c{ncomp}"
        print(f"\nEvaluating Configuration: {folder_name}", flush=True)

        # 1. EVALUATE BASELINE (data/)
        base_model_path = f"data/{folder_name}/{folder_name}_s0"
        base_csv = os.path.join(baseline_out_dir, f"inference_{folder_name}.csv")
        
        set_slo(slo, freq, knob, port=8000)
        print(f"  -> Running Baseline ({base_model_path})...", flush=True)
        
        params = {
            'log_dir': baseline_out_dir,
            'steps_per_epoch': steps_per_epoch,
            'budget': budget,
            'slo_latency': slo,
            'overrun_lim': overrun_lim,
            'mode': mode,
            'nconf': nconf,
            'ncomp': ncomp,
        }
        
        base_reward, base_util, base_breaches = evaluate_baseline(base_model_path, params, base_csv)

        # 2. EVALUATE A3C (a3c/)
        a3c_csv = os.path.join(a3c_out_dir, f"inference_{folder_name}.csv")
        set_slo(slo, freq, knob, port=8000)
        print(f"  -> Running A3C (a3c/{folder_name}/model/model1000.pt)...", flush=True)
        a3c_reward, a3c_util, a3c_breaches = evaluate_a3c(
            folder_name, nconf, ncomp, slo,
            steps_per_epoch, budget, overrun_lim, mode, a3c_csv
        )

        # 3. RECORD SUMMARY
        print(f"  [Summary {folder_name}]", flush=True)
        print(f"    Baseline | Reward: {base_reward} | Util: {base_util} | Breaches: {base_breaches}", flush=True)
        print(f"    A3C      | Reward: {a3c_reward:.2f} | Util: {a3c_util:.4f} | Breaches: {a3c_breaches}", flush=True)

        summary_records.append([
            nconf, ncomp,
            base_reward, a3c_reward,
            base_util, a3c_util,
            base_breaches, a3c_breaches
        ])

    # Save final consolidated comparison table
    summary_csv = os.path.join(root_out, 'quality_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Configs', 'Components',
            'Baseline_Avg_Reward', 'A3C_Avg_Reward',
            'Baseline_Avg_Util', 'A3C_Avg_Util',
            'Baseline_Breaches', 'A3C_Breaches'
        ])
        writer.writerows(summary_records)

    print(f"\nQuality benchmark complete! Consolidated summary saved to: {summary_csv}", flush=True)