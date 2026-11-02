# Load pretrained encoders and tokenize campaign prompts.
from dataclasses import dataclass
import json, os
from typing import Optional, Any
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    T5EncoderModel,
    T5Config,
    Qwen2Model,
    Qwen2Config,
    BartModel,
    BartConfig
)


@dataclass
class ModelConfig:
    """Architecture-specific model loading options."""
    name: str
    config_class: Optional[Any] = None
    model_class: Optional[Any] = None
    dropout_field: str = "dropout_rate"


class BartEncoderModel:
    @classmethod
    def from_pretrained(cls, model_name, config=None, **options):
        if config is not None:
            full_model = BartModel.from_pretrained(model_name, config=config, **options)
        else:
            full_model = BartModel.from_pretrained(model_name, **options)
        return full_model.encoder


MODEL_CONFIGS = {
    "t5-base": ModelConfig("t5-base", T5Config, T5EncoderModel),
    "t5-small": ModelConfig("t5-small", T5Config, T5EncoderModel),
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": ModelConfig(
        "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        T5Config,
        T5EncoderModel,
    ),
    "Qwen/Qwen2.5-0.5B": ModelConfig(
        "Qwen/Qwen2.5-0.5B", Qwen2Config, Qwen2Model, "attention_dropout"
    ),
    "facebook/bart-base": ModelConfig(
        "facebook/bart-base", BartConfig, BartEncoderModel, "attention_dropout"
    ),
}


def pretrained_options(model_name):
    revisions = json.loads(os.getenv('MALLAMAR_MODEL_REVISIONS', '{}'))
    revision = revisions.get(model_name)
    return {'revision': revision} if revision else {} # campaigns reuse the recorded commit


def get_model_and_tokenizer(model_name: str, device: str = 'cuda', model_dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, **pretrained_options(model_name)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_config := MODEL_CONFIGS.get(model_name):
        config = model_config.config_class.from_pretrained(model_name, **pretrained_options(model_name))

        if hasattr(config, model_config.dropout_field):
            setattr(config, model_config.dropout_field, 0)

        model = model_config.model_class.from_pretrained(
            model_name, config=config, **pretrained_options(model_name), **({"dtype": model_dtype} if model_dtype is not None else {})
        ).to(device)

    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **pretrained_options(model_name))
        possible_dropout_keys = [
            "dropout", "dropout_rate", "attn_dropout", "attention_dropout", "summary_last_dropout"
        ]
        for key in possible_dropout_keys:
            if hasattr(config, key):
                setattr(config, key, 0)

        model = AutoModel.from_pretrained(
            model_name, config=config, device_map=device, trust_remote_code=True, **pretrained_options(model_name),
            **({"dtype": model_dtype} if model_dtype is not None else {})
        )

    # Keep the full model through LoRA initialization to preserve campaign RNG order.
    return model, tokenizer


def get_tokens(
    texts,
    model_name="t5-base",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, **pretrained_options(model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    return torch.cat([encoded_input.input_ids, encoded_input.attention_mask], dim=1).cpu().numpy()
