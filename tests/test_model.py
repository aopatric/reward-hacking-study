import pytest
import torch

from src.config import ModelConfig
from src.model import capture_activations, generate, load_model_and_tokenizer, prompt_activations


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

    def test_base_model_has_no_trainable_adapter(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        assert not any("lora" in n for n, _ in model.named_parameters())
        assert not model.training

    def test_generation_reproducible_with_fixed_seed(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = [[{"role": "user", "content": "Say a random word."}]]

        torch.manual_seed(0)
        first, first_ids = generate(model, tokenizer, prompt, max_new_tokens=16, temperature=0.8)
        torch.manual_seed(0)
        second, second_ids = generate(model, tokenizer, prompt, max_new_tokens=16, temperature=0.8)

        assert first == second
        assert first_ids == second_ids

    def test_batched_generation_has_sampling_diversity(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = [[{"role": "user", "content": "Say a random word."}]]
        group_size = 4

        completions, token_ids = generate(
            model, tokenizer, prompt, num_return_sequences=group_size,
            max_new_tokens=16, temperature=1.0, top_p=1.0,
        )

        assert len(completions) == group_size
        assert len(token_ids) == group_size
        assert len(set(completions)) > 1  # catches an accidental fallback to greedy decoding

    def test_readonly_hook_leaves_generation_unchanged(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = [[{"role": "user", "content": "What is 2+2?"}]]

        torch.manual_seed(0)
        without_hook, _ = generate(model, tokenizer, prompt, max_new_tokens=16, do_sample=False)

        torch.manual_seed(0)
        with capture_activations(model, layers=[0, 5]):
            with_hook, _ = generate(model, tokenizer, prompt, max_new_tokens=16, do_sample=False)

        assert without_hook == with_hook

    def test_prompt_activations_shape_and_determinism(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = [{"role": "user", "content": "What is 2+2?"}]

        first = prompt_activations(model, tokenizer, prompt)
        second = prompt_activations(model, tokenizer, prompt)

        assert first.shape == (model.config.num_hidden_layers, model.config.hidden_size)
        assert first.dtype == torch.float32
        assert torch.equal(first, second)

    def test_pass_a_agrees_with_batched_prefill(self, model_and_tokenizer):
        """Pass A runs unbatched/unpadded; Pass B's prefill runs left-padded in a
        batch. They are not bit-identical -- this pins how far apart they are, so
        a later patching experiment knows which one it is injecting.
        """
        model, tokenizer = model_and_tokenizer
        prompts = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "Explain gradient descent in one paragraph."}],
        ]
        layer = 12

        canonical = prompt_activations(model, tokenizer, prompts[0])[layer]

        with capture_activations(model, layers=[layer]) as store:
            generate(model, tokenizer, prompts, max_new_tokens=4, do_sample=False)
        batched = store[layer][0][0, -1, :].float().cpu()

        cosine = torch.nn.functional.cosine_similarity(canonical, batched, dim=0)
        assert cosine > 0.99, f"cosine {cosine:.4f}, max abs diff {(canonical - batched).abs().max():.4f}"
