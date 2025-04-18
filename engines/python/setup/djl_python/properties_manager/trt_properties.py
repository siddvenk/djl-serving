#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
from pydantic import field_validator, ConfigDict
from typing import Optional

from djl_python.properties_manager.properties import Properties, RollingBatchEnum

TRT_SUPPORTED_ROLLING_BATCH_TYPES = [
    RollingBatchEnum.auto.value, RollingBatchEnum.trtllm.value,
    RollingBatchEnum.disable.value
]


class TensorRtLlmProperties(Properties):

    # in our implementation, we handle compilation/quantization ahead of time.
    # we do not expose build_config, quant_config, calib_config here as those get handled by
    # the compilation in trt_llm_partition.py. We do it this way so that users can completely build the complete
    # trt engine ahead of time via that script. If provided just a HF model id, then that script gets invoked,
    # does compilation/quantization and generates engines that will get loaded here. We are only exposing
    # runtime knobs here.

    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    dtype: str = 'auto'
    skip_tokenizer_init: bool = False
    tokenizer_revision: Optional[str] = None
    pipeline_parallel_degree: int = 1
    context_parallel_size: int = 1
    load_format: str = 'auto'
    enable_lora: bool = False
    max_lora_rank: Optional[int] = None
    max_loras: int = 4
    max_cpu_loras: int = 4,
    enable_prompt_adapter: bool = False
    max_prompt_adapter_token: int = 0
    # kv_cache_config
    enable_chunked_prefill: bool = False
    # decoding_config
    guided_decoding_backend: Optional[str] = None
    # logits_post_processor_map
    iter_stats_max_iterations: Optional[int] = None
    request_stats_max_iterations: Optional[int] = None
    embedding_parallel_mode: str = 'SHARDING_ALONG_VOCAB'
    auto_parallel: bool = False
    auto_parallel_world_size: int = 1
    moe_tensor_parallel_size: Optional[int] = None
    moe_expert_parallel_size: Optional[int] = None
    fast_build: bool = False
    # enable_build_cache
    # peft_cache_config
    # scheduler_config
    # speculative_config
    # batching_type
    normalize_log_probs: bool = False
    max_num_tokens: Optional[int] = None
    # extended_runtime_perf_knob

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    @field_validator('rolling_batch', mode='before')
    def validate_rolling_batch(cls, rolling_batch: str) -> str:
        rolling_batch = rolling_batch.lower()

        if rolling_batch not in TRT_SUPPORTED_ROLLING_BATCH_TYPES:
            raise ValueError(
                f"tensorrt llm only supports "
                f"rolling batch type {TRT_SUPPORTED_ROLLING_BATCH_TYPES}.")

        return rolling_batch

    def get_llm_kwargs(self) -> dict:
        return {
            "tokenizer": self.tokenizer,
            "tokenizer_mode": self.tokenizer_mode,
            "skip_tokenizer_init": self.skip_tokenizer_init,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_degree,
            "dtype": self.dtype,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "pipeline_parallel_size": self.pipeline_parallel_degree,
            "context_parallel_size": self.context_parallel_size,
            "load_format": self.load_format,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "guided_decoding_backend": self.guided_decoding_backend,
            "iter_stats_max_iterations": self.iter_stats_max_iterations,
            "request_stats_max_iterations": self.request_stats_max_iterations,
            "embedding_parallel_mode": self.embedding_parallel_mode,
            "moe_tensor_parallel_size": self.moe_tensor_parallel_size,
            "moe_expert_parallel_size": self.moe_expert_parallel_size,
            "normalize_log_probs": self.normalize_log_probs,
            "max_batch_size": self.max_rolling_batch_size,
            "max_num_tokens": self.max_num_tokens,
        }
