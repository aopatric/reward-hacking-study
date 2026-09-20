"""Component 2 -- model loading, LoRA, generation, and activation capture.

Original to this project (no porting/attribution needed).
"""

from collections import defaultdict
from contextlib import contextmanager

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from src.config import ModelConfig


def load_model_and_tokenizer(cfg: ModelConfig) -> tuple[PeftModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched decoder-only generation

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=getattr(torch, cfg.dtype),
    )
    if torch.cuda.is_available():
        base_model = base_model.to("cuda")

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    return model, tokenizer


def _decoder_layers(model) -> torch.nn.ModuleList:
    """Unwraps a PeftModel (if present) down to the transformers model's decoder
    layer list. Qwen2-architecture models expose this as `model.model.layers`.
    """
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    return base.model.layers


@contextmanager
def capture_activations(model, layers: list[int]):
    """Context manager capturing post-layer hidden states (residual stream)
    for the given decoder layer indices.

    Yields `store: dict[int, list[Tensor]]`. Each entry accumulates one tensor
    per forward call that layer participates in -- during a single teacher-
    forced pass that's exactly one call (full sequence), but during
    `model.generate()` with KV caching, each decoder layer's forward runs once
    per generated token, so the hook fires once per step and `store[layer]`
    ends up with one entry per step (the first covering the full prompt, the
    rest one new token each) rather than a single full-sequence tensor. The
    list accumulates every call rather than overwriting, so no data is
    silently dropped regardless of which usage pattern the caller is in --
    slicing/pooling across those entries is left to the caller.

    Hooks are always removed on exit, including on exception.
    """
    store: dict[int, list[torch.Tensor]] = defaultdict(list)
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            store[layer_idx].append(hidden_states.detach())

        return _hook

    try:
        decoder_layers = _decoder_layers(model)
        for layer_idx in layers:
            handles.append(decoder_layers[layer_idx].register_forward_hook(_make_hook(layer_idx)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def generate(
    model,
    tokenizer: PreTrainedTokenizer,
    prompts: list[list[dict[str, str]]],
    num_return_sequences: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> list[str]:
    """Batched chat-format generation. Returns `len(prompts) * num_return_sequences`
    decoded completion strings (prompt tokens stripped), grouped by prompt.
    """
    texts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
        model.device
    )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
