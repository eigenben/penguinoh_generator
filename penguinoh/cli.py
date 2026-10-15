"""Command line entry point: `penguinoh prepare | train | upload | generate`."""

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

from penguinoh.config import DEFAULT_LORA, HUB_REPO, OUTPUT_DIR, REPO_ROOT, RUNS_DIR, SAMPLERS, TRAIN_DIR, TrainConfig

DEFAULT_PROMPTS = REPO_ROOT / "prompts" / "eval.txt"


def default_run_name() -> str:
    return datetime.now().strftime("klein4b-%Y%m%d-%H%M")


def add_train_config_args(parser: argparse.ArgumentParser):
    """Expose every TrainConfig field as --kebab-case, defaulting to the dataclass value."""
    for field in dataclasses.fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        else:
            parser.add_argument(flag, type=field.type, default=field.default)


def cmd_prepare(args):
    from penguinoh.dataset import prepare

    prepare(Path(args.source), Path(args.train_dir), force=args.force, prune=args.prune)


def cmd_train(args):
    from penguinoh.train import train, upload

    config = TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)})
    output_dir = RUNS_DIR / args.run_name
    train(config, output_dir)
    if args.push:
        upload(output_dir, args.hub_repo)


def cmd_upload(args):
    from penguinoh.train import upload

    upload(Path(args.run_dir), args.hub_repo)


def cmd_generate(args):
    from penguinoh.generate import generate, load_pipeline, parse_lora, read_prompts, save_batch, set_lora

    prompts = args.prompt or read_prompts(Path(args.prompts_file))
    pipe = load_pipeline(args.sampler)
    results = {}
    for spec in args.lora:
        lora = parse_lora(spec)
        set_lora(pipe, lora, args.lora_scale)
        label = lora.label if lora else "no LoRA"
        results[label] = [
            generate(pipe, prompt, seed, args.sampler, args.steps, args.guidance, args.width, args.height)
            for prompt in prompts
            for seed in args.seeds
        ]
    save_batch(results, prompts, args.seeds, Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S"))


def main():
    parser = argparse.ArgumentParser(prog="penguinoh", description="Penguinoh image generator")
    sub = parser.add_subparsers(required=True)

    from penguinoh.dataset import DEFAULT_SOURCE_DIR

    p = sub.add_parser("prepare", help="Build data/train from website photos listed in metadata.jsonl")
    p.add_argument("--source", default=str(DEFAULT_SOURCE_DIR), help="Website image directory")
    p.add_argument("--train-dir", default=str(TRAIN_DIR))
    p.add_argument("--force", action="store_true", help="Re-process images that already exist")
    p.add_argument("--prune", action="store_true", help="Delete images not listed in the manifest")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="Train a LoRA on a local GPU into runs/<run-name>")
    p.add_argument("--run-name", default=default_run_name())
    p.add_argument("--push", action="store_true", help="Upload the result to the Hugging Face Hub")
    p.add_argument("--hub-repo", default=HUB_REPO)
    add_train_config_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("upload", help="Upload a finished run's LoRA weights to the Hub")
    p.add_argument("run_dir")
    p.add_argument("--hub-repo", default=HUB_REPO)
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("generate", help="Render prompts with one or more LoRAs (a grid when comparing)")
    p.add_argument("--prompt", action="append", help="Prompt or scene; repeatable. Defaults to prompts/eval.txt")
    p.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS))
    p.add_argument("--lora", action="append", help="none, a local run/checkpoint dir, or owner/repo[/checkpoint-N]")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--sampler", choices=SAMPLERS, default="base")
    p.add_argument("--steps", type=int, help="Override the sampler's step count")
    p.add_argument("--guidance", type=float, help="Override the sampler's guidance scale")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--out", default=str(OUTPUT_DIR))
    p.set_defaults(func=cmd_generate, lora=None)

    args = parser.parse_args()
    if args.func is cmd_generate and not args.lora:
        args.lora = [DEFAULT_LORA]
    args.func(args)


if __name__ == "__main__":
    main()
