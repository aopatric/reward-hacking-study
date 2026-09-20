import pytest
import torch

from src.config import ModelConfig
from src.model import capture_activations, generate, load_model_and_tokenizer


class _FakeDecoderLayer(torch.nn.Module):
    def forward(self, x):
        return (x + 1,)


class _FakeInner(torch.nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeDecoderLayer() for _ in range(n_layers)])


class _FakeModel(torch.nn.Module):
    """Mimics the `model.model.layers[i]` shape of a Qwen2-architecture model,
    without needing the real weights -- lets hook-lifecycle logic be tested
    without a GPU or a model download.
    """

    def __init__(self):
        super().__init__()
        self.model = _FakeInner()


def test_hook_removed_even_on_exception():
    fake_model = _FakeModel()
    layer = fake_model.model.layers[0]
    assert len(layer._forward_hooks) == 0

    with pytest.raises(RuntimeError):
        with capture_activations(fake_model, layers=[0]):
            raise RuntimeError("boom")

    assert len(layer._forward_hooks) == 0


def test_capture_activations_accumulates_one_entry_per_call():
    fake_model = _FakeModel()
    x = torch.randn(2, 3, 4)

    with capture_activations(fake_model, layers=[0, 2]) as store:
        fake_model.model.layers[0](x)
        fake_model.model.layers[0](x)  # simulates a second forward call (e.g. a KV-cache step)
        fake_model.model.layers[2](x)

    assert len(store[0]) == 2
    assert len(store[2]) == 1
    assert torch.equal(store[0][0], x + 1)


@pytest.mark.gpu
class TestWithRealModel:
    @pytest.fixture(scope="class")
    @classmethod
    def model_and_tokenizer(cls):
        return load_model_and_tokenizer(ModelConfig())

    def test_lora_params_trainable_base_frozen(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n]
        base_params = [(n, p) for n, p in model.named_parameters() if "lora" not in n]

        assert lora_params
        assert all(p.requires_grad for _, p in lora_params)
        assert all(not p.requires_grad for _, p in base_params)

    def test_generation_reproducible_with_fixed_seed(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = [[{"role": "user", "content": "Say a random word."}]]

        torch.manual_seed(0)
        first = generate(model, tokenizer, prompt, max_new_tokens=16, temperature=0.8)
        torch.manual_seed(0)
        second = generate(model, tokenizer, prompt, max_new_tokens=16, temperature=0.8)

        assert first == second

    def test_batched_generation_has_sampling_diversity(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = [[{"role": "user", "content": "Say a random word."}]]
        group_size = 4

        completions = generate(
            model, tokenizer, prompt, num_return_sequences=group_size,
            max_new_tokens=16, temperature=1.0, top_p=1.0,
        )

        assert len(completions) == group_size
        assert len(set(completions)) > 1  # catches an accidental fallback to greedy decoding

    def test_readonly_hook_leaves_generation_unchanged(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = [[{"role": "user", "content": "What is 2+2?"}]]

        torch.manual_seed(0)
        without_hook = generate(model, tokenizer, prompt, max_new_tokens=16, do_sample=False)

        torch.manual_seed(0)
        with capture_activations(model, layers=[0, 5]):
            with_hook = generate(model, tokenizer, prompt, max_new_tokens=16, do_sample=False)

        assert without_hook == with_hook
