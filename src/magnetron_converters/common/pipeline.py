# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import ctypes
import json
import os
import glob
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from safetensors import safe_open

from magnetron import SnapshotWriter, dtype

JsonValue = int | float | str | bool | None
TargetMap = dict[str, tuple[tuple[int, ...], dtype.DType]]
ManifestRow = tuple[str, tuple[int, ...], str]

_TORCH_FOR_MAG: dict[dtype.DType, torch.dtype] = {
    dtype.float16: torch.float16,
    dtype.bfloat16: torch.bfloat16,
    dtype.float32: torch.float32,
}

_MAG_FOR_STR: dict[str, dtype.DType] = {
    'float16': dtype.float16,
    'bfloat16': dtype.bfloat16,
    'float32': dtype.float32,
}


def mag_to_torch_dtype(mag_dtype: dtype.DType) -> torch.dtype:
    return _TORCH_FOR_MAG[mag_dtype]


def mag_dtype_from_str(dtype_str: str) -> dtype.DType:
    return _MAG_FOR_STR[dtype_str]


def iter_safetensor_shards(repo_dir: str) -> list[str]:
    index_path = os.path.join(repo_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path, encoding='utf-8') as f:
            index = json.load(f)
        shards = sorted(set(index['weight_map'].values()))
        return [os.path.join(repo_dir, s) for s in shards]
    shards = sorted(glob.glob(os.path.join(repo_dir, 'model-*.safetensors')))
    if shards:
        return shards
    single = os.path.join(repo_dir, 'model.safetensors')
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError('No safetensors weights found in repo snapshot.')


@contextmanager
def no_weight_init() -> Iterator[None]:
    import magnetron.nn.layers as layers

    original = layers.inplace_init
    layers.inplace_init = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        layers.inplace_init = original


def torch_buffer(tensor: torch.Tensor) -> memoryview:
    tensor = tensor.contiguous()
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes == 0:
        return memoryview(b'')
    raw = (ctypes.c_char * nbytes).from_address(tensor.data_ptr())
    return memoryview(raw).cast('B')


def stream_convert(
    snap_file: str,
    *,
    targets: TargetMap,
    shards: list[str],
    hf_key_for: Callable[[str], str],
    mag_dtype: dtype.DType,
    torch_dtype: torch.dtype,
    meta: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
    on_missing: Callable[[str, tuple[int, ...], dtype.DType], torch.Tensor | None] | None = None,
    verbose: bool = True,
) -> list[ManifestRow]:
    aliases = dict(aliases or {})
    manifest: list[ManifestRow] = []
    owned = {k: v for k, v in targets.items() if k not in aliases}
    remaining = dict(owned)

    with SnapshotWriter(snap_file) as snap:
        snap.meta = dict(meta)
        for key, (shape, target_dtype) in owned.items():
            snap.declare(key, target_dtype, list(shape))
        for key, target in aliases.items():
            shape, _ = targets[key]
            snap.alias(key, target, list(shape))
            manifest.append((key, tuple(shape), targets[key][1].short_name))
            if verbose:
                print(f'Aliasing {key} -> {target} (shares bytes, costs nothing)')
        snap.commit_layout()

        for shard_path in shards:
            if not remaining:
                break
            with safe_open(shard_path, framework='pt', device='cpu') as shard:
                available = set(shard.keys())
                for key in list(remaining.keys()):
                    hf_key = hf_key_for(key)
                    if hf_key not in available:
                        continue
                    shape, target_dtype = remaining[key]
                    src = shard.get_tensor(hf_key)
                    if tuple(src.shape) != tuple(shape):
                        raise RuntimeError(f'Shape mismatch for {key}: {tuple(src.shape)} != {tuple(shape)}')
                    if verbose:
                        print(f'Converting {hf_key} -> {key} shape={tuple(src.shape)} dtype={target_dtype.short_name}')
                    cast_to = torch_dtype if target_dtype == mag_dtype else mag_to_torch_dtype(target_dtype)
                    out = src.to(cast_to).contiguous()
                    snap.write(key, torch_buffer(out))
                    manifest.append((key, tuple(shape), target_dtype.short_name))
                    del src, out
                    remaining.pop(key)

        if remaining:
            for key in list(remaining.keys()):
                shape, target_dtype = remaining[key]
                replacement = on_missing(key, tuple(shape), target_dtype) if on_missing else None
                if replacement is None:
                    raise KeyError(f'Missing HF weights for magnetron keys: {sorted(remaining.keys())}')
                snap.write(key, torch_buffer(replacement.contiguous()))
                manifest.append((key, tuple(shape), target_dtype.short_name))
                remaining.pop(key)

    return manifest


def write_model_card(
    path: str,
    *,
    repo: str,
    snap_file: str,
    mag_dtype: dtype.DType,
    cfg: object,
    tensor_rows: list[ManifestRow],
    config_heading: str,
) -> None:
    tensor_rows = sorted(tensor_rows, key=lambda x: x[0])
    model_name = repo.split('/')[-1]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {model_name} Magnetron Snapshot\n\n')
        f.write(f'This repository contains a Magnetron snapshot converted from the original Hugging Face model `{repo}`.\n\n')
        f.write('The snapshot is intended for inference with the Magnetron runtime. ')
        f.write(f'All convertible tensors are stored using `{mag_dtype.short_name}` where applicable.\n\n')
        f.write('## Model details\n\n')
        f.write(f'- **Source model:** `{repo}`\n')
        f.write(f'- **Snapshot file:** `{snap_file}`\n')
        f.write(f'- **Magnetron dtype mode:** `{mag_dtype.short_name}`\n')
        f.write(f'- **Tensor count:** `{len(tensor_rows)}`\n\n')
        f.write(f'## {config_heading}\n\n')
        f.write('| Field | Value |\n')
        f.write('|---|---:|\n')
        for k, v in vars(cfg).items():
            f.write(f'| `{k}` | `{v}` |\n')
        f.write('\n')
        f.write('## Tensor manifest\n\n')
        f.write('| Name | Shape | DType |\n')
        f.write('|---|---:|---|\n')
        for name, shape, dt in tensor_rows:
            shape_s = 'x'.join(str(x) for x in shape)
            f.write(f'| `{name}` | `{shape_s}` | `{dt}` |\n')


def config_meta(cfg: object) -> dict[str, JsonValue]:
    return {k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool, type(None)))}
