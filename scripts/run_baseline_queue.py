#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_METHODS = ["diffusion_soup", "dmm", "maxfusion", "anp", "clean_finetune"]

DEFAULT_CKPTS = {
    "clean": "./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.0_BOX_14-HAT_clean",
    "bd_box14_hat": "./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT_bd_box14_hat",
    "bd_box11_cat": "./res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_11-CAT_bd_box11_cat",
}

DEFAULT_SCENARIOS = {
    "s1_hat": {
        "ckpts": ["clean", "bd_box14_hat"],
        "trigger": "BOX_14",
        "target": "HAT",
    },
    "s1_cat": {
        "ckpts": ["clean", "bd_box11_cat"],
        "trigger": "BOX_11",
        "target": "CAT",
    },
    "s2_hat": {
        "ckpts": ["bd_box14_hat", "bd_box11_cat"],
        "trigger": "BOX_14",
        "target": "HAT",
    },
    "s2_cat": {
        "ckpts": ["bd_box14_hat", "bd_box11_cat"],
        "trigger": "BOX_11",
        "target": "CAT",
    },
}


@dataclass
class Job:
    name: str
    method: str
    ckpts: str
    trigger: str
    target: str


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run merge-defense baseline jobs as a GPU queue.")
    parser.add_argument("--gpus", default="2", help="Comma-separated GPU ids.")
    parser.add_argument("--python", default="/home1/zhln/envs/baddiffusion/bin/python")
    parser.add_argument("--prefix", default="baseline", help="Output prefix, e.g. save or final.")
    parser.add_argument("--scenarios", default="s1_hat,s1_cat,s2_hat,s2_cat")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--sample_n", type=int, default=1024)
    parser.add_argument("--eval_max_batch", type=int, default=64)
    parser.add_argument("--num_inference_steps", type=int, default=200)
    parser.add_argument("--alphas", default="0.5")
    parser.add_argument("--model_weights", default="0.5,0.5")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--skip_fid", dest="skip_fid", action="store_true", default=True)
    parser.add_argument("--no_skip_fid", dest="skip_fid", action="store_false")
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
    parser.add_argument("--no_save_model", dest="save_model", action="store_false")
    parser.add_argument("--force_resample", dest="force_resample", action="store_true", default=True)
    parser.add_argument("--no_force_resample", dest="force_resample", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_jobs(args):
    jobs = []
    for scenario_name in parse_csv(args.scenarios):
        if scenario_name not in DEFAULT_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        scenario = DEFAULT_SCENARIOS[scenario_name]
        ckpts = ",".join(DEFAULT_CKPTS[key] for key in scenario["ckpts"])
        for method in parse_csv(args.methods):
            jobs.append(
                Job(
                    name=f"{args.prefix}_{scenario_name}_{method}",
                    method=method,
                    ckpts=ckpts,
                    trigger=scenario["trigger"],
                    target=scenario["target"],
                )
            )
    return jobs


def summary_path(job):
    return Path(job.name) / "merge_summary.json"


def is_done(job):
    path = summary_path(job)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(data) and "asr" in data[0]


def command_for(job, args):
    cmd = [
        args.python,
        "merge.py",
        "--method",
        job.method,
        "--model_ckpts",
        job.ckpts,
        "--model_weights",
        args.model_weights,
        "--alphas",
        args.alphas,
        "--output_dir",
        job.name,
        "--dataset",
        args.dataset,
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
    ]
    if args.skip_fid:
        cmd.append("--skip_fid")
    if args.save_model:
        cmd.append("--save_model")
    if args.force_resample:
        cmd.append("--force_resample")
    return cmd


def main():
    args = parse_args()
    gpus = parse_csv(args.gpus)
    if not gpus:
        raise SystemExit("No GPU ids provided.")

    log_dir = Path(args.log_dir or f"logs/{args.prefix}")
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = [job for job in build_jobs(args) if args.force or not is_done(job)]
    running = {}

    print(f"[QUEUE] pending={len(queue)} gpus={gpus} log_dir={log_dir}", flush=True)
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
