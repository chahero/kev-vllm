"""Experimental Kev pointer readout inside the vLLM GPU worker.

One causal row per question, original Kev delimiter token IDs, no generation.
Returns [number of options, 1] calibrated probabilities via /pooling.
"""
from pathlib import Path

import torch
from torch import nn
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM


class KevPointerPooler(Pooler):
    def __init__(self, config):
        super().__init__()
        self.pooling = AllPool()
        self.q = nn.Linear(config.hidden_size, config.kev_head_dim, dtype=torch.float32)
        self.k = nn.Linear(config.hidden_size, config.kev_head_dim, dtype=torch.float32)
        self.scale = config.kev_head_dim ** -0.5
        self.temperature = config.kev_temperature
        self.option_end_id = config.kev_option_end_id
        self.decide_id = config.kev_decide_id

    def get_supported_tasks(self):
        return {"token_embed"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states, pooling_metadata):
        chunks = self.pooling(hidden_states, pooling_metadata)
        token_rows = pooling_metadata.get_prompt_token_ids_cpu()
        results = []
        for h, ids in zip(chunks, token_rows):
            if h is None:
                results.append(None)
                continue
            ids = ids[:h.shape[0]]
            opts = (ids == self.option_end_id).nonzero(as_tuple=True)[0]
            decide = (ids == self.decide_id).nonzero(as_tuple=True)[0]
            if not len(opts) or len(decide) != 1:
                # vLLM 0.30 profiles with zeros and warms kernels with range(N).
                # These synthetic inputs cannot be produced by the Kev encoder.
                is_profile = pooling_metadata.prompt_token_ids is None and not bool(ids.any())
                is_warmup = torch.equal(ids, torch.arange(len(ids), dtype=ids.dtype))
                if is_profile or is_warmup:
                    results.append(torch.zeros((1, 1), device=h.device, dtype=torch.float32))
                    continue
                raise ValueError("Expected one Kev question row with options and one decide token")
            if int(decide[0]) != len(ids) - 1 or bool((opts >= decide[0]).any()):
                raise ValueError("Kev decide token must follow every option and end the row")
            h = h.float()
            scores = self.k(h[opts.to(h.device)]) @ self.q(h[int(decide[0])])
            probs = torch.softmax(scores * self.scale / self.temperature, dim=-1)
            results.append(probs[:, None])
        return results


@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
class KevQwen3_5ForPooling(Qwen3_5ForCausalLM):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pooler = KevPointerPooler(vllm_config.model_config.hf_text_config)
        self.head_path = Path(vllm_config.model_config.model) / "head.pt"

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        saved = torch.load(self.head_path, map_location="cpu", weights_only=True)
        for name, tensor in saved["head"].items():
            layer, attr = name.split(".")
            parameter = getattr(getattr(self.pooler, layer), attr)
            with torch.no_grad():
                parameter.copy_(tensor)
            loaded.add(f"pooler.{name}")
        return loaded
