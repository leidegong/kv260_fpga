"""Model configurations used by the Step 3 virtual prototype and performance model.

Qwen3 values are copied from the official config.json files on Hugging Face
(Qwen/Qwen3-1.7B, Qwen/Qwen3-0.6B). LLaMA configs are only used to calibrate the
performance model against the numbers reported by DATE'25 / Hummingbird.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCfg:
    name: str
    hidden: int
    inter: int
    layers: int
    n_q: int            # attention heads
    n_kv: int           # key/value heads (GQA)
    head_dim: int
    vocab: int
    tied: bool          # tie_word_embeddings: lm_head.weight is embed_tokens.weight
    qk_norm: bool       # Qwen3 applies RMSNorm to every q/k head before RoPE
    rope_theta: float
    eps: float = 1e-6

    @property
    def q_dim(self):
        return self.n_q * self.head_dim

    @property
    def kv_dim(self):
        return self.n_kv * self.head_dim

    @property
    def gqa(self):
        return self.n_q // self.n_kv

    def linears(self):
        """(name, out_features, in_features) of one decoder layer, in decode streaming order."""
        return [
            ("q_proj", self.q_dim, self.hidden),
            ("k_proj", self.kv_dim, self.hidden),
            ("v_proj", self.kv_dim, self.hidden),
            ("o_proj", self.hidden, self.q_dim),
            ("gate_proj", self.inter, self.hidden),
            ("up_proj", self.inter, self.hidden),
            ("down_proj", self.hidden, self.inter),
        ]

    def layer_params(self):
        return sum(o * i for _, o, i in self.linears())

    def total_params(self):
        emb = self.vocab * self.hidden
        return self.layers * self.layer_params() + emb * (1 if self.tied else 2)


QWEN3_1_7B = ModelCfg("Qwen3-1.7B", 2048, 6144, 28, 16, 8, 128, 151936, True, True, 1e6)
QWEN3_0_6B = ModelCfg("Qwen3-0.6B", 1024, 3072, 28, 16, 8, 128, 151936, True, True, 1e6)
LLAMA2_7B = ModelCfg("LLaMA2-7B", 4096, 11008, 32, 32, 32, 128, 32000, False, False, 1e4, 1e-5)
LLAMA3_8B = ModelCfg("LLaMA3-8B", 4096, 14336, 32, 32, 8, 128, 128256, False, False, 5e5, 1e-5)

# Small Qwen3-shaped model for self-tests. q_dim != hidden (like Qwen3-0.6B), GQA = 2
# (like Qwen3-1.7B), vocab not a multiple of the page/group sizes.
TINY = ModelCfg("tiny-qwen3", 256, 768, 3, 4, 2, 128, 1000, True, True, 1e6)

MODELS = {m.name: m for m in (QWEN3_1_7B, QWEN3_0_6B, LLAMA2_7B, LLAMA3_8B, TINY)}
