# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import math
from dataclasses import dataclass, fields
from typing import Any

from magnetron import Tensor, dtype

from magnetron_converters import common


# TODO: this must be pulled from magnetron-models
@dataclass
class Config:
    repo_id: str = 'openai/whisper-base'
    vocab_size: int = 51865
    num_mel_bins: int = 80
    d_model: int = 512
    encoder_layers: int = 6
    encoder_attention_heads: int = 8
    encoder_ffn_dim: int = 2048
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 2048
    max_source_positions: int = 1500
    max_target_positions: int = 448

    @classmethod
    def from_hf_config(cls, repo_id: str, raw: dict[str, Any]) -> Config:
        return cls(repo_id=repo_id, **{f.name: raw[f.name] for f in fields(cls) if f.name != 'repo_id' and f.name in raw})
###

_MODEL_PREFIX: str = 'model.'
_KNOWN_MODELS: tuple[str, ...] = (
    'openai/whisper-tiny',
    'openai/whisper-base',
    'openai/whisper-small',
    'openai/whisper-medium',
    'openai/whisper-large-v3',
    'openai/whisper-large-v3-turbo',
)


def _mag_key_for(hf_key: str) -> str | None:
    if hf_key.startswith(_MODEL_PREFIX):
        return hf_key[len(_MODEL_PREFIX) :]
    return None


# Slaney mel scale
_HZ_PER_MEL: float = 200.0 / 3.0  # Linear region: 66.67 Hz per mel
_LOG_START_HZ: float = 1000.0  # Linear below, logarithmic above
_LOG_START_MEL: float = _LOG_START_HZ / _HZ_PER_MEL  # = 15 mel
_LOG_STEP: float = math.log(6.4) / 27.0  # 27 mels per 6.4x in frequency


def _mel_to_hz(m: Tensor) -> Tensor:
    return Tensor.where(m < _LOG_START_MEL, m * _HZ_PER_MEL, _LOG_START_HZ * ((m - _LOG_START_MEL) * _LOG_STEP).exp())


def _compute_mel_filters(n_mels: int, sample_rate: int, n_fft: int) -> Tensor:
    top = _LOG_START_MEL + math.log(sample_rate / 2.0 / _LOG_START_HZ) / _LOG_STEP  # hz_to_mel(Nyquist), hz_to_mel(0) = 0
    hz = _mel_to_hz(Tensor.linspace(0.0, top, n_mels + 2, dtype=dtype.float32)).reshape(-1, 1)
    fdiff = hz[1:] - hz[:-1]
    fft_freqs = Tensor.arange(0, n_fft // 2 + 1, dtype=dtype.float32).reshape(1, -1) * (sample_rate / n_fft)
    ramps = hz - fft_freqs
    lower = -ramps[:-2] / fdiff[:-1]
    upper = ramps[2:] / fdiff[1:]
    enorm = 2.0 / (hz[2:] - hz[:-2])  # Slaney norm: equal area per filter
    return lower.min(upper).clamp_min(0.0) * enorm


def _mel_filters(preproc: dict[str, Any], cfg: Config) -> Tensor:
    n_mels: int = preproc.get('feature_size', cfg.num_mel_bins)
    if n_mels != cfg.num_mel_bins:
        raise ValueError(f'processor_config.json has {n_mels} mel bins but the model expects {cfg.num_mel_bins}')
    shipped = preproc.get('mel_filters')
    if shipped is not None:  # Older repos ship the exact reference filterbank, prefer it
        return Tensor(shipped, dtype=dtype.float32)
    return _compute_mel_filters(n_mels, preproc.get('sampling_rate', 16000), preproc.get('n_fft', 400))


def _validate(plan: list[common.TensorPlan], cfg: Config) -> None:
    common.check_shapes(
        plan,
        {
            'encoder.conv1.weight': (cfg.d_model, cfg.num_mel_bins, 3),
            'encoder.embed_positions.weight': (cfg.max_source_positions, cfg.d_model),
            'decoder.embed_tokens.weight': (cfg.vocab_size, cfg.d_model),
            'decoder.embed_positions.weight': (cfg.max_target_positions, cfg.d_model),
            f'encoder.layers.{cfg.encoder_layers - 1}.fc1.weight': (cfg.encoder_ffn_dim, cfg.d_model),
            f'decoder.layers.{cfg.decoder_layers - 1}.fc1.weight': (cfg.decoder_ffn_dim, cfg.d_model),
            f'decoder.layers.{cfg.decoder_layers - 1}.encoder_attn.q_proj.weight': (cfg.d_model, cfg.d_model),
        },
    )


def convert(
    repo: str,
    mag_dtype: dtype.DType,
    out: str | None = None,
    *,
    write_model_card: bool = True,
    model_card_path: str = 'model_card.md',
    card_only: bool = False,
) -> str:
    repo_dir: str = common.download_repo(repo, allow_patterns=['*.json', 'model.safetensors'])
    hf_config = common.load_hf_config(repo_dir)
    if hf_config.get('model_type') != 'whisper':
        raise ValueError(f'{repo} is not a Whisper model (model_type={hf_config.get("model_type")})')
    cfg = Config.from_hf_config(repo, hf_config)
    plan = common.plan_tensors(repo_dir, mag_key_for=_mag_key_for, dtype_for=common.dtype_policy(mag_dtype))
    _validate(plan, cfg)
    preproc = common.load_hf_config(repo_dir, 'preprocessor_config.json')
    mel_filters = _mel_filters(preproc, cfg)
    return common.convert_repo(
        repo,
        repo_dir,
        plan,
        mag_dtype=mag_dtype,
        arch='whisper',
        model='whisper',
        cfg=cfg,
        config_title='Whisper configuration',
        out=out,
        write_model_card=write_model_card,
        model_card_path=model_card_path,
        card_only=card_only,
        extra_metadata={
            'gen_cfg': common.load_hf_config(repo_dir, 'generation_config.json'),
            'pp_cfg': {k: v for k, v in preproc.items() if k != 'mel_filters'},
        },
        extra_tensors={'mel_filters': mel_filters},
    )


def main() -> None:
    parser = common.build_arg_parser(
        'Convert a Hugging Face Whisper model to the Magnetron snapshot format', default_model='openai/whisper-base', known_models=_KNOWN_MODELS
    )
    args = parser.parse_args()
    convert(
        args.model,
        common.mag_dtype_from_str(args.dtype),
        args.out,
        write_model_card=args.model_card,
        model_card_path=args.model_card_path,
        card_only=args.card_only,
    )


if __name__ == '__main__':
    main()
