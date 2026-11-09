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
    hf_token: str | None = None,
    proxy: str | None = None,
) -> None:
    """Download file over HTTP with resume support, proxy support, and progress tracking."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    downloaded = 0
    if temp_path.exists() and temp_path.stat().st_size > 0:
        downloaded = temp_path.stat().st_size

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    proxy_url = proxy or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("all_proxy")
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    # 1. Try downloading via requests (best for proxies, SOCKS, redirects, and resume)
    try:
        import requests

        req_headers = dict(headers)
        if downloaded > 0:
            req_headers["Range"] = f"bytes={downloaded}-"

        with requests.get(url, headers=req_headers, proxies=proxies, stream=True, timeout=(20, 120)) as resp:
            if resp.status_code == 416:
                # Range not satisfiable (file changed or invalid .part) -> reset from 0
                downloaded = 0
                temp_path.unlink(missing_ok=True)
                resp = requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(20, 120))

            resp.raise_for_status()

            total_len = resp.headers.get("Content-Length")
            total = int(total_len) + downloaded if total_len else expected_size or 0

            mode = "ab" if downloaded > 0 and resp.status_code == 206 else "wb"
            if mode == "wb":
                downloaded = 0

            with open(temp_path, mode) as out:
                for chunk in resp.iter_content(chunk_size=128 * 1024):
                    if chunk:
                        out.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total > 0:
                            progress_callback(downloaded, total)

        if temp_path.exists():
            temp_path.replace(destination)
        return
    except ImportError:
        pass
    except Exception as req_exc:
        # Fall back to urllib if requests failed
        pass

    # 2. Fallback to urllib.request
    req_headers = dict(headers)
    if downloaded > 0:
        req_headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=req_headers)
    try:
        handlers = []
        if proxies:
            handlers.append(urllib.request.ProxyHandler(proxies))
        opener = urllib.request.build_opener(*handlers)

        with opener.open(req, timeout=60) as resp:
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
    except urllib.error.HTTPError as e:
        if e.code == 416 and temp_path.exists():
            temp_path.unlink(missing_ok=True)
            return _download_http_fallback(url, destination, expected_size, progress_callback, hf_token, proxy)
        raise
    except Exception:
        raise


def download_file(
    repo_id: str,
    mf: ModelFile,
    target_dir: Path,
    force: bool = False,
    quiet: bool = False,
    hf_token: str | None = None,
    proxy: str | None = None,
) -> Path:
    """Download a single model file using huggingface_hub with HTTP fallback."""
    rel_path = mf.target_path or mf.filename
    dest = target_dir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if valid non-empty file already exists
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        if mf.size_bytes and dest.stat().st_size != mf.size_bytes:
            pass  # Size mismatch, re-download
        elif mf.sha256:
            if compute_sha256(dest) == mf.sha256:
                if not quiet:
                    print(f"  [OK] {rel_path} already downloaded and verified ({dest.stat().st_size:,} bytes).")
                return dest
        else:
            if not quiet:
                print(f"  [OK] {rel_path} already exists ({dest.stat().st_size:,} bytes).")
            return dest
    elif dest.is_file() and dest.stat().st_size == 0:
        # Clean up 0-byte corrupted/aborted file
        dest.unlink(missing_ok=True)

    subfolder = mf.subfolder or ""
    source_filename = mf.filename
    effective_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

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
            token=effective_token,
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
            if pbar is not None:
                pbar.total = total
                pbar.n = downloaded
                pbar.refresh()

        try:
            _download_http_fallback(
                url,
                dest,
                expected_size=mf.size_bytes,
                progress_callback=cb,
                hf_token=effective_token,
                proxy=proxy,
            )
        finally:
            if pbar is not None:
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
    hf_token: str | None = None,
    proxy: str | None = None,
) -> bool:
    """Download all files for a specific model."""
    if isinstance(model, str):
        model = get_model(model)

    target_dir = model.get_local_path(models_dir)

    # Check if the model is already installed with valid non-empty files
    if not force and model.is_installed(models_dir):
        if not quiet:
            print(f"\n=== Model '{model.id}' is already fully downloaded on disk. Skipping download. ===")
        return True

    target_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"\n=== Downloading model '{model.id}' ({model.name}) ===")
        print(f"  Target: {target_dir}")
        print(f"  Size: ~{model.size_mb:.1f} MB | Engine: {model.engine_name}")

    for mf in model.files:
        try:
            download_file(
                mf.repo_id or model.repo_id,
                mf,
                target_dir,
                force=force,
                quiet=quiet,
                hf_token=hf_token,
                proxy=proxy,
            )
        except Exception as exc:
            if not quiet:
                print(f"  [Error] Failed to download {mf.filename}: {exc}")
            return False

    # Check for default voice symlinks if downloading pocket or piper
    voices_dir = target_dir.parent / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    default_voice = voices_dir / "default.wav"
    if not default_voice.exists():
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
    hf_token: str | None = None,
    proxy: str | None = None,
) -> dict[str, bool]:
    """Download all models in a profile."""
    models = get_profile_models(profile)
    if not quiet:
        print(f"Starting download for profile '{profile}' ({len(models)} models)")

    results: dict[str, bool] = {}
    for m in models:
        success = download_model(
            m,
            models_dir=models_dir,
            force=force,
            quiet=quiet,
            hf_token=hf_token,
            proxy=proxy,
        )
        results[m.id] = success
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="nava-download",
        description="Nava Persian Speech Model Downloader (Hugging Face / Releases)",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional positional target: profile (all, minimal, realtime, high-accuracy), model ID, or 'list'",
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
    parser.add_argument(
        "--proxy",
        type=str,
        default=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("all_proxy"),
        help="HTTP/HTTPS/SOCKS5 proxy URL (e.g. http://127.0.0.1:10809 or socks5://127.0.0.1:10808)",
    )
    parser.add_argument(
        "--hf-token",
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face User Access Token (from https://huggingface.co/settings/tokens)",
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir) if args.models_dir else None

    # Propagate proxy and HF token to environment for libraries
    if args.proxy:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ[k] = args.proxy

    if args.hf_token:
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            os.environ[k] = args.hf_token

    # Handle positional target convenience (e.g. `nava-download all` or `nava-download whisper-large-v3`)
    profile = args.profile
    model = args.model
    if args.target:
        if args.target in ("minimal", "realtime", "high-accuracy", "all"):
            profile = args.target
        elif args.target in ("list", "--list"):
            args.list = True
        else:
            model = args.target

    if args.list:
        models = list_models(models_dir)
        print(f"\n{'Model ID':<32} {'Type':<6} {'Size(MB)':<10} {'Installed':<10} {'Profiles'}")
        print("-" * 80)
        for m in models:
            inst = "[YES]" if m["installed"] else "[NO]"
            profs = ", ".join(m["profiles"])
            print(f"{m['id']:<32} {m['type']:<6} {m['size_mb']:<10.1f} {inst:<10} {profs}")
        return 0

    if model:
        success = download_model(
            model,
            models_dir=models_dir,
            force=args.force,
            quiet=args.quiet,
            hf_token=args.hf_token,
            proxy=args.proxy,
        )
        return 0 if success else 1

    if profile:
        results = download_profile(
            profile,
            models_dir=models_dir,
            force=args.force,
            quiet=args.quiet,
            hf_token=args.hf_token,
            proxy=args.proxy,
        )
        failed = [mid for mid, ok in results.items() if not ok]
        if failed:
            print(f"Warning: Failed models: {failed}")
            return 1
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

