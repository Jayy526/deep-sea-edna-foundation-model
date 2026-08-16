"""Genomic foundation-model encoder (GenomeOcean).

MODEL IDENTITY -- PAPER-DERIVED
-------------------------------
TaxDistill's teacher branch is GenomeOcean with a frozen backbone. We use the
same published checkpoint: ``pGenomeOcean/GenomeOcean-500M``. This is the
actual model from the paper, not a substitute.

VERIFIED COMPATIBILITY WITH 151 bp READS
----------------------------------------
Checked against the published checkpoint before running anything:

  architecture         MistralForCausalLM (causal LM, 14 layers)
  hidden_size          1536      -> embedding dimension
  vocabulary           4096 BPE tokens over the ACGT alphabet
  max input length     1024 tokens (~5 kbp per the model card)
  weights              model.safetensors, bfloat16

A 151 bp read tokenises to roughly 30-40 BPE tokens, which is ~3% of the
model's 1024-token limit. The model therefore accepts our reads directly, with
no padding to a target length and no truncation. This is checked at runtime by
``verify_compatibility()`` rather than assumed.

WHAT THIS DOES NOT CLAIM
------------------------
GenomeOcean was pretrained on assembled genomic sequence. Feeding it 151 bp
reads is IN-SPEC with respect to input length but OUT-OF-DISTRIBUTION with
respect to the context length it was trained to exploit. That is precisely the
question this experiment tests; it is not something we can assume away. See
docs/METHODOLOGY.md.

POOLING -- IMPLEMENTATION DECISION
----------------------------------
The paper states the teacher "extracts deep semantic features" and projects
them through a classification head, but does not specify the pooling operation.
We default to attention-masked mean pooling over the final hidden layer, which
is the standard choice for causal-LM sequence embeddings and is unbiased with
respect to read position. ``last`` and ``max`` pooling are also implemented and
selectable from the config so the choice is auditable, not baked in.

AMBIGUOUS BASES
---------------
The BPE vocabulary covers ACGT only; an ``N`` would become ``[UNK]``. Quality
control resolves this upstream (see preprocessing/quality_control.py), so the
encoder never sees an ambiguous base under the default configuration. The
runtime check ``unk_token_count`` confirms this rather than trusting it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from models.encoder import SequenceEncoder

DEFAULT_MODEL_ID = "pGenomeOcean/GenomeOcean-500M"
VALID_POOLING = {"mean", "last", "max"}


class FoundationModelEncoder(SequenceEncoder):
    """Frozen genomic foundation model used as a fixed feature extractor."""

    kind = "foundation"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = "main",
        pooling: str = "mean",
        dtype: str = "bfloat16",
        device: str = "auto",
        attn_implementation: str = "sdpa",
        max_tokens: int = 1024,
        normalize: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if pooling not in VALID_POOLING:
            raise ValueError(f"pooling must be one of {sorted(VALID_POOLING)}")

        self.model_id = model_id
        self.revision = revision
        self.pooling = pooling
        self.max_tokens = max_tokens
        self.normalize = normalize
        self.name = model_id.split("/")[-1].lower().replace("-", "_")

        self.device = self._resolve_device(device, torch)
        self.torch_dtype = self._resolve_dtype(dtype, self.device, torch)
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # transformers >=5 renamed ``torch_dtype`` to ``dtype``; support both so the
        # pipeline is not pinned to one transformers major version.
        common = dict(
            revision=revision,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=self.torch_dtype, **common
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, **common
            )
        self.attn_implementation = getattr(
            self.model.config, "_attn_implementation", attn_implementation
        )

        # FROZEN BACKBONE (paper-derived): no parameter of the foundation model
        # is ever updated. Inference only, gradients disabled everywhere.
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(self.device)

        self._backbone = self.model.model  # skip the LM head; we want hidden states
        self._hidden_size = int(self.model.config.hidden_size)
        self._param_total = sum(p.numel() for p in self.model.parameters())
        self._param_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self._token_lengths: list[int] = []
        self._unk_tokens = 0
        self._truncated = 0

    # --- Setup helpers -----------------------------------------------------

    @staticmethod
    def _resolve_device(device: str, torch) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(dtype: str, device: str, torch):
        if device == "cpu":
            # bf16/fp16 matmul on CPU is slow or unsupported; fp32 is the honest fallback.
            return torch.float32
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"dtype must be one of {sorted(mapping)}")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            return torch.float16
        return mapping[dtype]

    # --- Compatibility verification ----------------------------------------

    def verify_compatibility(self, sample_sequences: Sequence[str]) -> dict[str, Any]:
        """Measure, on real reads, that this model can actually take our input.

        Returns the measured token statistics. Raises if the reads do not fit.
        """
        encoded = self.tokenizer(list(sample_sequences), add_special_tokens=True)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        unk_id = self.tokenizer.unk_token_id
        n_unk = (
            sum(ids.count(unk_id) for ids in encoded["input_ids"])
            if unk_id is not None
            else 0
        )
        max_len = max(lengths)
        model_limit = min(self.max_tokens, int(self.model.config.max_position_embeddings))
        report = {
            "model_id": self.model_id,
            "n_sample_reads": len(sample_sequences),
            "bp_per_read_min": min(len(s) for s in sample_sequences),
            "bp_per_read_max": max(len(s) for s in sample_sequences),
            "tokens_per_read_min": min(lengths),
            "tokens_per_read_max": max_len,
            "tokens_per_read_mean": round(sum(lengths) / len(lengths), 3),
            "bp_per_token_mean": round(
                sum(len(s) for s in sample_sequences) / sum(lengths), 3
            ),
            "model_token_limit": model_limit,
            "fraction_of_limit_used": round(max_len / model_limit, 5),
            "fits_without_truncation": max_len <= model_limit,
            "unk_tokens_in_sample": n_unk,
            "padding_applied_to_reach_target_length": False,
        }
        if not report["fits_without_truncation"]:
            raise RuntimeError(
                f"{self.model_id} cannot accept these reads without truncation: "
                f"{max_len} tokens > limit {model_limit}."
            )
        return report

    # --- Encoding ----------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._hidden_size

    def encode_batch(self, sequences: Sequence[str]) -> np.ndarray:
        torch = self._torch
        batch = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            add_special_tokens=True,
        )
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]

        lengths = mask.sum(dim=1)
        self._token_lengths.extend(lengths.tolist())
        if self.tokenizer.unk_token_id is not None:
            self._unk_tokens += int(
                ((input_ids == self.tokenizer.unk_token_id) & mask.bool()).sum()
            )
        self._truncated += int((lengths >= self.max_tokens).sum())

        input_ids = input_ids.to(self.device)
        mask = mask.to(self.device)

        with torch.inference_mode():
            hidden = self._backbone(
                input_ids=input_ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
            pooled = self._pool(hidden, mask)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.float().cpu().numpy()

    def _pool(self, hidden, mask):
        torch = self._torch
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "mean":
            summed = (hidden * mask_f).sum(dim=1)
            return summed / mask_f.sum(dim=1).clamp(min=1)
        if self.pooling == "last":
            # padding_side='left', so the final column is always a real token.
            return hidden[:, -1, :]
        if self.pooling == "max":
            return (hidden.masked_fill(mask_f == 0, torch.finfo(hidden.dtype).min)).max(
                dim=1
            ).values
        raise ValueError(f"Unhandled pooling: {self.pooling}")

    # --- Reporting ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        lengths = np.array(self._token_lengths) if self._token_lengths else np.array([])
        return {
            "name": self.name,
            "kind": self.kind,
            "label": "PAPER-DERIVED (same checkpoint as TaxDistill's teacher branch)",
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture": self.model.config.architectures,
            "embedding_dim": self.embedding_dim,
            "num_hidden_layers": self.model.config.num_hidden_layers,
            "vocab_size": self.model.config.vocab_size,
            "max_position_embeddings": self.model.config.max_position_embeddings,
            "max_tokens_used": self.max_tokens,
            "pooling": self.pooling,
            "normalized": self.normalize,
            "model_parameters_total": self._param_total,
            "model_parameters_trainable": self._param_trainable,
            "model_parameters_frozen": self._param_total - self._param_trainable,
            "backbone_frozen": True,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "attn_implementation": self.attn_implementation,
            "observed_tokens_per_read": {
                "n": int(lengths.size),
                "min": int(lengths.min()) if lengths.size else None,
                "max": int(lengths.max()) if lengths.size else None,
                "mean": round(float(lengths.mean()), 3) if lengths.size else None,
            },
            "unk_tokens_encountered": self._unk_tokens,
            "reads_truncated": self._truncated,
        }

    def input_handling(self) -> dict[str, Any]:
        return {
            "transformation": (
                "Each QC-passed read (<=151 bp) is uppercased, BPE-tokenised by the "
                "model's own 4096-token genomic tokenizer, and passed to the frozen "
                "backbone at its true length."
            ),
            "padding": (
                "Left padding within a batch only, so that variable-length reads can "
                "share a tensor. Padding tokens are excluded from pooling by the "
                "attention mask and contribute nothing to the embedding. No read is "
                "ever padded toward a target sequence length."
            ),
            "truncation": f"Only if a read exceeds {self.max_tokens} tokens; measured count reported as 'reads_truncated'.",
            "concatenation": "None. Reads are never joined; no genomic context is invented.",
            "pooling": f"{self.pooling} over the final hidden layer, attention-masked",
        }

    def close(self) -> None:
        try:
            del self._backbone, self.model
            self._torch.cuda.empty_cache()
        except Exception:
            pass


def build_encoder(cfg: dict) -> SequenceEncoder:
    """Construct the encoder named by ``cfg['encoder']['backend']``."""
    enc_cfg = cfg["encoder"]
    backend = enc_cfg["backend"]
    if backend in ("genomeocean", "foundation"):
        return FoundationModelEncoder(
            model_id=enc_cfg["model_id"],
            revision=enc_cfg.get("revision", "main"),
            pooling=enc_cfg.get("pooling", "mean"),
            dtype=enc_cfg.get("dtype", "bfloat16"),
            device=enc_cfg.get("device", "auto"),
            attn_implementation=enc_cfg.get("attn_implementation", "sdpa"),
            max_tokens=enc_cfg.get("max_tokens", 1024),
            normalize=enc_cfg.get("normalize", False),
        )
    raise ValueError(f"Unknown encoder backend: {backend!r}")


def build_baseline(cfg: dict) -> SequenceEncoder:
    """Construct the baseline encoder named by ``cfg['baseline']['backend']``."""
    from models.baseline_encoder import KmerBaselineEncoder

    base_cfg = cfg["baseline"]
    if base_cfg["backend"] in ("kmer_tnf", "kmer", "baseline"):
        return KmerBaselineEncoder(
            k=base_cfg.get("k", 4),
            canonical=base_cfg.get("canonical", True),
            include_gc=base_cfg.get("include_gc", True),
            include_length=base_cfg.get("include_length", False),
        )
    raise ValueError(f"Unknown baseline backend: {base_cfg['backend']!r}")
