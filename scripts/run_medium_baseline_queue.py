#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


METHODS = ["diffusion_soup", "dmm", "maxfusion", "anp", "clean_finetune"]

CLEAN = "./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT_clean"
BD_BOX14_HAT = "./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat"
BD_BOX11_CAT = "./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat"


@dataclass
class Job:
    name: str
    method: str
    ckpts: str
    trigger: str
    target: str


def build_jobs():
    jobs = []
    scenarios = [
        ("s1_box14", f"{CLEAN},{BD_BOX14_HAT}", "BOX_14", "HAT"),
        ("s1_box11", f"{CLEAN},{BD_BOX11_CAT}", "BOX_11", "CAT"),
        ("s2_hat", f"{BD_BOX14_HAT},{BD_BOX11_CAT}", "BOX_14", "HAT"),
        ("s2_cat", f"{BD_BOX14_HAT},{BD_BOX11_CAT}", "BOX_11", "CAT"),
    ]
    for scenario, ckpts, trigger, target in scenarios:
        for method in METHODS:
            jobs.append(
                Job(
                    name=f"merge_medium_{scenario}_{method}",
                    method=method,
                    ckpts=ckpts,
                    trigger=trigger,
                    target=target,
                )
            )
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description="Queue medium-size merge baseline runs.")
    parser.add_argument("--gpus", default="0,1,2,4", help="Comma-separated GPU ids to use.")
    parser.add_argument("--python", default="/home1/zhln/envs/baddiffusion/bin/python")
    parser.add_argument("--sample_n", type=int, default=256)
    parser.add_argument("--eval_max_batch", type=int, default=64)
    parser.add_argument("--num_inference_steps", type=int, default=200)
    parser.add_argument("--log_dir", default="logs/merge_medium")
    parser.add_argument("--force", action="store_true", help="Rerun jobs even if merge_summary.json exists.")
    parser.add_argument("--save_model", action="store_true", help="Pass --save_model to merge.py.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def is_done(job: Job):
    summary = Path(job.name) / "merge_summary.json"
    if not summary.exists():
        return False
    try:
        data = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return False
    return bool(data) and "asr" in data[0]


def command_for(job: Job, args):
    cmd = [
        args.python,
        "merge.py",
        "--method",
        job.method,
        "--model_ckpts",
        job.ckpts,
        "--model_weights",
        "0.5,0.5",
        "--alphas",
        "0.5",
        "--output_dir",
        job.name,
        "--dataset",
        "CIFAR10",
        "--trigger",
        job.trigger,
        "--target",
        job.target,
        "--sample_n",
        str(args.sample_n),
        "--eval_max_batch",
        str(args.eval_max_batch),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--skip_fid",
        "--force_resample",
    ]
    if args.save_model:
        cmd.append("--save_model")
    return cmd


def main():
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided.")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = [job for job in build_jobs() if args.force or not is_done(job)]
    running = {}

    print(f"[QUEUE] {len(queue)} jobs pending, GPUs={gpus}", flush=True)
    if args.dry_run:
        for job in queue:
            print(" ".join(command_for(job, args)))
        return

    while queue or running:
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            job = queue.pop(0)
            log_path = log_dir / f"{job.name}.log"
            log_fh = log_path.open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["WANDB_MODE"] = "disabled"
            print(f"[START] gpu={gpu} job={job.name} log={log_path}", flush=True)
            proc = subprocess.Popen(command_for(job, args), stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (job, proc, log_fh)

        time.sleep(10)
        for gpu, (job, proc, log_fh) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            log_fh.close()
            status = "DONE" if ret == 0 and is_done(job) else f"FAILED ret={ret}"
            print(f"[{status}] gpu={gpu} job={job.name}", flush=True)
            del running[gpu]


if __name__ == "__main__":
    main()
