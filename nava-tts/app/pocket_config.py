"""Localize hf:// references in the pocket model.yaml to files on disk.

pocket_tts resolves hf:// paths through the HuggingFace cache, which does not
exist in the offline container (no egress network, read-only model volume).
Every hf:// reference must exist under the local pocket directory; because
/models is mounted read-only, the patched config is written to a temp file.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_HF_REF = re.compile(r"hf://[^/\s\"']+/[^/\s\"']+/([^\s\"']+)")

_ALIASES: dict[str, tuple[str, ...]] = {
    "tokenizer.model": ("tokenizer.model", "tokenizer_ph.model"),
    "tokenizer_ph.model": ("tokenizer_ph.model", "tokenizer.model"),
    "farsi.yaml": ("farsi.yaml", "model.yaml"),
    "model.yaml": ("model.yaml", "farsi.yaml"),
}


def localize_pocket_config(config: Path, pocket_dir: Path) -> Path:
    """Return a config path whose hf:// URLs point at local files in pocket_dir.

    Each hf://<owner>/<repo>/<file>[@revision] is rewritten to pocket_dir/<file>;
    the referenced file must exist, otherwise startup fails with a clear message.
    A config without hf:// references is returned unchanged.
    """
    text = config.read_text(encoding="utf-8")

    def _local(match: "re.Match[str]") -> str:
        filename = match.group(1).split("@", 1)[0]
        for cand in _ALIASES.get(filename, (filename,)):
            local = pocket_dir / cand
            if local.is_file():
                return str(local)
        raise RuntimeError(f"Pocket model file missing: {pocket_dir / filename} — run scripts/download_models.sh")

    patched = _HF_REF.sub(_local, text)
    if patched == text:
        return config
    target = Path(tempfile.gettempdir()) / f"nava-pocket-{config.stem}-{os.getpid()}.yaml"
    target.write_text(patched, encoding="utf-8")
    return target
