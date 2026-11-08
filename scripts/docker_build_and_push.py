#!/usr/bin/env python3
"""Build and optionally push Nava Docker images to Docker Hub."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVICES = {
    "stt": [("nava-stt", ROOT / "nava-stt" / "Dockerfile", ROOT)],
    "tts": [("nava-tts", ROOT / "nava-tts" / "Dockerfile", ROOT)],
    "server": [("nava", ROOT / "Dockerfile", ROOT)],
}


def run_cmd(cmd: list[str]) -> None:
    print(f"\n==> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Error: Command failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and optionally push Nava Docker images to Docker Hub"
    )
    parser.add_argument(
        "-u",
        "--username",
        default=os.getenv("DOCKERHUB_USERNAME", ""),
        help="Docker Hub username/organization (or set DOCKERHUB_USERNAME env var)",
    )
    parser.add_argument(
        "-t",
        "--tag",
        default="latest",
        help="Docker image tag (default: latest)",
    )
    parser.add_argument(
        "-s",
        "--service",
        choices=["all", "stt", "tts", "server"],
        default="all",
        help="Service to build/push (default: all)",
    )
    parser.add_argument(
        "-p",
        "--push",
        action="store_true",
        help="Push images to Docker Hub after building",
    )
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Target platform architecture (default: linux/amd64)",
    )

    args = parser.parse_args()

    username = args.username.strip().lower()
    if not username:
        print("Error: Docker Hub username is required.", file=sys.stderr)
        print("Provide via --username <username> or DOCKERHUB_USERNAME environment variable.", file=sys.stderr)
        print("Example: python3 scripts/docker_build_and_push.py --username mhosseinifar8", file=sys.stderr)
        return 1

    targets: list[tuple[str, Path, Path]] = []
    if args.service == "all":
        for s_list in SERVICES.values():
            targets.extend(s_list)
    else:
        targets.extend(SERVICES[args.service])

    print("======================================================")
    print(" Nava Docker Builder")
    print(f" User:     {username}")
    print(f" Tag:      {args.tag}")
    print(f" Platform: {args.platform}")
    print(f" Service:  {args.service}")
    print(f" Push:     {args.push}")
    print("======================================================")

    for name, dockerfile, context in targets:
        full_tag = f"{username}/{name}:{args.tag}"
        cmd = [
            "docker", "build",
            "--platform", args.platform,
            "-t", full_tag,
            "-f", str(dockerfile),
            str(context)
        ]
        run_cmd(cmd)

        if name == "nava":
            # Tag alias nava-server as well
            alias_tag = f"{username}/nava-server:{args.tag}"
            run_cmd(["docker", "tag", full_tag, alias_tag])

        if args.push:
            run_cmd(["docker", "push", full_tag])
            if name == "nava":
                alias_tag = f"{username}/nava-server:{args.tag}"
                run_cmd(["docker", "push", alias_tag])

    print("\n✓ Docker build operation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

