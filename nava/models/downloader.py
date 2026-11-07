"""Nava Model Downloader.

Automates downloading and verifying CPU-optimized Persian STT, TTS, and G2P models
from Hugging Face and GitHub Releases with checksum verification, progress bars,
and resume support.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from .registry import (MODELS, ModelFile, ModelInfo, ProfileName, get_model,
                       get_profile_models, list_models)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


def compute_sha256(filepath: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_http_fallback(
    url: str,
    destination: Path,
    expected_size: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Download file over HTTP with resume support and progress tracking."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    downloaded = 0
    headers = {}
    if temp_path.exists():
        downloaded = temp_path.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_length = resp.headers.get("Content-Length")
            total = int(content_length) + downloaded if content_length else expected_size or 0

            mode = "ab" if downloaded > 0 and resp.status == 206 else "wb"
            if mode == "wb":
                downloaded = 0

            with open(temp_path, mode) as out:
                chunk_size = 64 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total > 0:
                        progress_callback(downloaded, total)

        if temp_path.exists():
            temp_path.replace(destination)
    except Exception:
        # Keep .part for resume, re-raise error
        raise


def download_file(
    repo_id: str,
    mf: ModelFile,
    target_dir: Path,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Download a single model file using huggingface_hub with HTTP fallback."""
    rel_path = mf.target_path or mf.filename
    dest = target_dir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if mf.size_bytes and dest.stat().st_size != mf.size_bytes:
            pass  # Size mismatch, re-download
        elif mf.sha256:
            if compute_sha256(dest) == mf.sha256:
                if not quiet:
                    print(f"  [OK] {rel_path} already downloaded and verified.")
                return dest
        else:
            if not quiet:
                print(f"  [OK] {rel_path} already exists.")
            return dest

    subfolder = mf.subfolder or ""
    source_filename = mf.filename

    # Attempt download via huggingface_hub first
    hf_success = False
    try:
        from huggingface_hub import hf_hub_download

        if not quiet:
            print(f"  Downloading {rel_path} from HF {repo_id}...")
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=source_filename,
            subfolder=subfolder if subfolder else None,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        if Path(downloaded_file).resolve() != dest.resolve():
            # In case local_dir preserved subfolder structure
            downloaded_p = Path(downloaded_file)
            if downloaded_p.exists() and downloaded_p != dest:
                shutil.move(str(downloaded_p), str(dest))
        hf_success = dest.exists()
    except Exception as e:
        if not quiet:
            print(f"  huggingface_hub download failed ({e}), falling back to direct HTTP...")

    if not hf_success:
        url_file = f"{subfolder}/{source_filename}".lstrip("/")
        url = f"https://huggingface.co/{repo_id}/resolve/main/{url_file}"
        if not quiet:
            print(f"  Downloading {url} -> {dest}...")

        pbar = None
        if not quiet and tqdm:
            pbar = tqdm(unit="B", unit_scale=True, desc=rel_path)

        def cb(downloaded: int, total: int):
            if pbar:
                pbar.total = total
                pbar.n = downloaded
                pbar.refresh()

        try:
            _download_http_fallback(url, dest, expected_size=mf.size_bytes, progress_callback=cb)
        finally:
            if pbar:
                pbar.close()

    # Post-download verification
    if mf.sha256 and dest.exists():
        actual_hash = compute_sha256(dest)
        if actual_hash != mf.sha256:
            dest.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {dest}: expected {mf.sha256}, got {actual_hash}")

    if not quiet:
        print(f"  [Done] {rel_path}")
    return dest


def download_model(
    model: ModelInfo | str,
    models_dir: Path | str | None = None,
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """Download all files for a specific model."""
    if isinstance(model, str):
        model = get_model(model)

    target_dir = model.get_local_path(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"\n=== Downloading model '{model.id}' ({model.name}) ===")
        print(f"  Target: {target_dir}")
        print(f"  Size: ~{model.size_mb:.1f} MB | Engine: {model.engine_name}")

    for mf in model.files:
        try:
            download_file(model.repo_id, mf, target_dir, force=force, quiet=quiet)
        except Exception as exc:
            if not quiet:
                print(f"  [Error] Failed to download {mf.filename}: {exc}")
            return False

    # Check for default voice symlinks if downloading pocket or piper
    voices_dir = target_dir.parent / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    default_voice = voices_dir / "default.wav"
    if not default_voice.exists():
        # Look for any available wav
        for wav in voices_dir.glob("*.wav"):
            if wav.name != "default.wav":
                try:
                    default_voice.symlink_to(wav.name)
                    break
                except OSError:
                    pass

    if not quiet:
        print(f"=== Successfully prepared '{model.id}' ===")
    return True


def download_profile(
    profile: ProfileName,
    models_dir: Path | str | None = None,
    force: bool = False,
    quiet: bool = False,
) -> dict[str, bool]:
    """Download all models in a profile."""
    models = get_profile_models(profile)
    if not quiet:
        print(f"Starting download for profile '{profile}' ({len(models)} models)")

    results: dict[str, bool] = {}
    for m in models:
        success = download_model(m, models_dir=models_dir, force=force, quiet=quiet)
        results[m.id] = success
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="nava-download",
        description="Nava Persian Speech Model Downloader (Hugging Face / Releases)",
    )
    parser.add_argument(
        "--profile",
        choices=["minimal", "realtime", "high-accuracy", "all"],
        default=None,
        help="Download models by pre-configured profile",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Download a specific model ID (e.g. shenava-koochik-ctc, piper-mana-lca)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Target models directory (defaults to $NAVA_MODELS_DIR or ./models)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered models and their download status",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress output",
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir) if args.models_dir else None

    if args.list:
        models = list_models(models_dir)
        print(f"\n{'Model ID':<32} {'Type':<6} {'Size(MB)':<10} {'Installed':<10} {'Profiles'}")
        print("-" * 80)
        for m in models:
            inst = "[YES]" if m["installed"] else "[NO]"
            profs = ", ".join(m["profiles"])
            print(f"{m['id']:<32} {m['type']:<6} {m['size_mb']:<10.1f} {inst:<10} {profs}")
        return 0

    if args.model:
        success = download_model(args.model, models_dir=models_dir, force=args.force, quiet=args.quiet)
        return 0 if success else 1

    if args.profile:
        results = download_profile(args.profile, models_dir=models_dir, force=args.force, quiet=args.quiet)
        failed = [mid for mid, ok in results.items() if not ok]
        if failed:
            print(f"Warning: Failed models: {failed}")
            return 1
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

