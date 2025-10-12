#!/usr/bin/env python3
"""
Simple benchmarking harness that runs `scripts/online_finetune_ppo.py` over a small grid of vectorization configs,
parses its per-epoch timing output, and prints a CSV summary.

This runs the training script as a subprocess with a single short epoch per config (timesteps = n_steps * num_envs)
so it completes quickly and reports steps/sec and timing breakdown.

Usage:
    python3 scripts/vec_benchmark.py

You can pass comma-separated lists for the parameters, e.g.: --num-envs 8,16 --n-steps 2048,4096

Note: run this in your project's venv (the same env you use for training).
"""

import argparse
import subprocess
import shlex
import re
import itertools
import csv
from pathlib import Path

EP_LINE_RE = re.compile(r"Ep\s+\d+\s+\|\s+Steps\s+[0-9/]+\s+\|\s+Time\s+\d+s\s+\|\s+Epoch time\s+([0-9.]+)s\s+\|\s+([0-9.]+)\s+steps/s")
TIMING_RE = re.compile(r"timing breakdown \(s\): env_step=([0-9.]+) transfer=([0-9.]+) model=([0-9.]+)")


def run_config(python='python3', script='scripts/online_finetune_ppo.py', num_envs=8, n_steps=2048, minibatch=256, update_epochs=8, device='cuda', use_amp=True, timeout=600):
    timesteps = n_steps * num_envs  # one epoch
    cmd = [python, script,
           '--timesteps', str(timesteps),
           '--n_steps', str(n_steps),
           '--minibatch_size', str(minibatch),
           '--update_epochs', str(update_epochs),
           '--device', device,
           '--num-envs', str(num_envs),
    ]
    if use_amp:
        cmd.append('--use-amp')

    # Run the command and capture output
    print('Running:', ' '.join(shlex.quote(x) for x in cmd))
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'timeout', 'output': ''}

    out = proc.stdout

    # parse the last epoch line and timing breakdown
    ep_match = None
    timing_match = None
    for m in EP_LINE_RE.finditer(out):
        ep_match = m
    for m in TIMING_RE.finditer(out):
        timing_match = m

    result = {'ok': proc.returncode == 0, 'returncode': proc.returncode, 'output': out,
              'steps_per_sec': None, 'epoch_time': None, 'env_step': None, 'transfer': None, 'model': None}

    if ep_match:
        result['epoch_time'] = float(ep_match.group(1))
        result['steps_per_sec'] = float(ep_match.group(2))
    if timing_match:
        result['env_step'] = float(timing_match.group(1))
        result['transfer'] = float(timing_match.group(2))
        result['model'] = float(timing_match.group(3))

    return result


def parse_list(s, cast=int):
    if isinstance(s, (list, tuple)):
        return [cast(x) for x in s]
    if s is None or s == '':
        return []
    return [cast(x) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, default='scripts/online_finetune_ppo.py')
    parser.add_argument('--python', type=str, default='python3')
    parser.add_argument('--num-envs', type=str, default='8,16,32')
    parser.add_argument('--n-steps', type=str, default='2048,4096')
    parser.add_argument('--minibatch', type=str, default='256,512')
    parser.add_argument('--update-epochs', type=str, default='8,16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--out-csv', type=str, default='vec_benchmark_results.csv')
    args = parser.parse_args()

    num_envs_list = parse_list(args.num_envs, int)
    n_steps_list = parse_list(args.n_steps, int)
    minibatch_list = parse_list(args.minibatch, int)
    update_epochs_list = parse_list(args.update_epochs, int)

    combos = list(itertools.product(num_envs_list, n_steps_list, minibatch_list, update_epochs_list))
    print('Running', len(combos), 'config(s)')

    results = []
    for (num_envs, n_steps, minibatch, update_epochs) in combos:
        res = run_config(python=args.python, script=args.script, num_envs=num_envs, n_steps=n_steps, minibatch=minibatch, update_epochs=update_epochs, device=args.device, use_amp=args.use_amp, timeout=args.timeout)
        row = {'num_envs': num_envs, 'n_steps': n_steps, 'minibatch': minibatch, 'update_epochs': update_epochs}
        row.update(res)
        results.append(row)
        # write incremental
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print('\nSummary:')
    for r in results:
        print(f"num_envs={r['num_envs']} n_steps={r['n_steps']} minibatch={r['minibatch']} epochs={r['update_epochs']} -> steps/s={r.get('steps_per_sec')} epoch_time={r.get('epoch_time')} env_step={r.get('env_step')} model={r.get('model')}")


if __name__ == '__main__':
    main()
