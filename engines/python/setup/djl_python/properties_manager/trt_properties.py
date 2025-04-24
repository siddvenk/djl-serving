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
import json
import os
import logging
from pydantic import ConfigDict
from typing import Optional

from tensorrt_llm.auto_parallel import infer_cluster_config
from tensorrt_llm.commands.build import parse_arguments
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode
from tensorrt_llm.plugin import PluginConfig, add_plugin_argument
from tensorrt_llm.llmapi import BuildConfig, QuantConfig, CalibConfig, QuantAlgo, KvCacheConfig

from djl_python.properties_manager.properties import Properties, RollingBatchEnum

logger = logging.getLogger(__name__)


class TensorRtLlmProperties(Properties):

    # https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/tensorrt_llm/llmapi/llm_args.py#L679
    # Configs that are not exposed currently:
    # - LoRA configs,
    # - prompt adapter configs
    # - cp_config
    # - batched_logits_processor
    # - workspace
    # - peft cache config
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    skip_tokenizer_init: bool = False
    dtype: str = 'auto'
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    moe_tensor_parallel_size: Optional[int] = None
    moe_expert_parallel_size: Optional[int] = None
    enable_attention_dp: bool = False
    auto_parallel: bool = False
    auto_parallel_world_size: Optional[int] = None
    load_format: str = 'auto'
    enable_chunked_prefill: bool = False
    guided_decoding_backend: Optional[str] = None
    iter_stats_max_iterations: Optional[int] = None
    request_stats_max_iterations: Optional[int] = None
    embedding_parallel_mode: str = 'SHARDING_ALONG_VOCAB'
    fast_build: bool = False
    # different default! allows for faster loading on worker restart
    enable_build_cache: bool = True
    batching_type: Optional[None] = None
    normalize_log_probs: bool = False
    gather_generation_logits: bool = False
    extended_runtime_perf_knob_config: Optional[None] = None
    max_batch_size: Optional[int] = None
    max_input_len: int = 1024
    max_seq_len: Optional[int] = None
    max_beam_width: int = 1
    max_num_tokens: Optional[int] = None
    backend: Optional[str] = None

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    def get_quant_config(self) -> Optional[QuantConfig]:
        quant_config = {}
        if "quant_algo" in self.__pydantic_extra__:
            quant_config["quant_algo"] = QuantAlgo(
                self.__pydantic_extra__["quant_algo"].upper())
        if "kv_cache_quant_algo" in self.__pydantic_extra__:
            quant_config["kv_cache_quant_algo"] = QuantAlgo(
                self.__pydantic_extra__["kv_cache_quant_algo"].upper())
        if "group_size" in self.__pydantic_extra__:
            quant_config["group_size"] = int(
                self.__pydantic_extra__["group_size"])
        if "smoothquant_val" in self.__pydantic_extra__:
            quant_config["smoothquant_val"] = float(
                self.__pydantic_extra__["smoothquant_val"])
        if "clamp_val" in self.__pydantic_extra__:
            quant_config["clamp_val"] = json.loads(
                self.__pydantic_extra__["clamp_val"])
        if "use_meta_recipe" in self.__pydantic_extra__:
            quant_config["use_meta_recipe"] = self.__pydantic_extra__[
                "use_meta_recipe"].lower() == "true"
        if "has_zero_point" in self.__pydantic_extra__:
            quant_config["has_zero_point"] = self.__pydantic_extra__[
                "has_zero_point"].lower() == "true"
        if "pre_quant_scales" in self.__pydantic_extra__:
            quant_config["pre_quant_scales"] = self.__pydantic_extra__[
                "pre_quant_scales"].lower() == "true"
        if "exclude_modules" in self.__pydantic_extra__:
            quant_config["exclude_modules"] = json.loads(
                self.__pydantic_extra__["exclude_modules"])
        if quant_config:
            return QuantConfig(**quant_config)
        return None

    def get_calib_config(self) -> Optional[CalibConfig]:
        calib_config = {}
        if "device" in self.__pydantic_extra__:
            calib_config["device"] = self.__pydantic_extra__["device"]
        if "calib_dataset" in self.__pydantic_extra__:
            calib_config["calib_dataset"] = self.__pydantic_extra__[
                "calib_dataset"]
        if "calib_batches" in self.__pydantic_extra__:
            calib_config["calib_batches"] = int(
                self.__pydantic_extra__["calib_batches"])
        if "calib_batch_size" in self.__pydantic_extra__:
            calib_config["calib_batch_size"] = int(
                self.__pydantic_extra__["calib_batch_size"])
        if "calib_max_seq_length" in self.__pydantic_extra__:
            calib_config["calib_max_seq_length"] = int(
                self.__pydantic_extra__["calib_max_seq_length"])
        if "random_seed" in self.__pydantic_extra__:
            calib_config["random_seed"] = int(
                self.__pydantic_extra__["random_seed"])
        if "tokenizer_max_seq_length" in self.__pydantic_extra__:
            calib_config["tokenizer_max_seq_length"] = int(
                self.__pydantic_extra__["tokenizer_max_seq_length"])
        if calib_config:
            return CalibConfig(**calib_config)
        return None

    def get_kv_cache_config(self) -> Optional[KvCacheConfig]:
        kv_cache_config = {}
        if "enable_block_reuse" in self.__pydantic_extra__:
            kv_cache_config["enable_block_reuse"] = self.__pydantic_extra__[
                "enable_block_reuse"].lower() == "true"
        if "max_tokens" in self.__pydantic_extra__:
            kv_cache_config["max_tokens"] = int(
                self.__pydantic_extra__["max_tokens"])
        if "max_attention_window" in self.__pydantic_extra__:
            kv_cache_config["max_attention_window"] = json.loads(
                self.__pydantic_extra__["max_attention_window"])
        if "sink_token_length" in self.__pydantic_extra__:
            kv_cache_config["sink_token_length"] = int(
                self.__pydantic_extra__["sink_token_length"])
        if "free_gpu_memory_fraction" in self.__pydantic_extra__:
            kv_cache_config["free_gpu_memory_fraction"] = float(
                self.__pydantic_extra__["free_gpu_memory_fraction"])
        if "host_cache_size" in self.__pydantic_extra__:
            kv_cache_config["host_cache_size"] = int(
                self.__pydantic_extra__["host_cache_size"])
        if "onboard_blocks" in self.__pydantic_extra__:
            kv_cache_config["onboard_blocks"] = self.__pydantic_extra__[
                "onboard_blocks"].lower() == "true"
        if "cross_kv_cache_fraction" in self.__pydantic_extra__:
            kv_cache_config["cross_kv_cache_fraction"] = float(
                self.__pydantic_extra__["cross_kv_cache_fraction"])
        if "secondary_offload_min_priority" in self.__pydantic_extra__:
            kv_cache_config["secondary_offload_min_priority"] = int(
                self.__pydantic_extra__["secondary_offload_min_priority"])
        if "event_buffer_max_size" in self.__pydantic_extra__:
            kv_cache_config["event_buffer_max_size"] = int(
                self.__pydantic_extra__["event_buffer_max_size"])
        if kv_cache_config:
            return KvCacheConfig(**kv_cache_config)
        return None

    def get_build_config(self) -> Optional[BuildConfig]:
        # https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/tensorrt_llm/commands/build.py
        build_args = self.__pydantic_extra__.copy()
        build_args["max_batch_size"] = self.max_rolling_batch_size
        trtllm_args = []
        for k, v in build_args.items():
            trtllm_args.append(f'--{k}')
            trtllm_args.append(f'{v}')
        parser = parse_arguments()
        args, unknown = parser.parse_known_args(args=trtllm_args)
        logger.info(
            f"[LMI] The following args will be passed to the build_config for TRTLLM: {args}"
        )
        if hasattr(args, 'gather_generation_logits'):
            logger.warning(
                'Option --gather_generation_logits is deprecated, a build flag is not required anymore. Use --output_generation_logits at runtime instead.'
            )

        if args.gather_all_token_logits:
            args.gather_context_logits = True
            args.gather_generation_logits = True
        if args.gather_context_logits and args.max_draft_len > 0:
            raise RuntimeError(
                "Gather context logits is not support with draft len > 0. "
                "If want to get the accepted tokens' logits from target model, please just enable gather_generation_logits"
            )

        if hasattr(args, 'paged_kv_cache'):
            logger.warning(
                'Option --paged_kv_cache is deprecated, use --kv_cache_type=paged/disabled instead.'
            )

        plugin_config = PluginConfig.from_arguments(args)
        plugin_config.validate()
        if self.fast_build:
            plugin_config.manage_weights = True

        speculative_decoding_mode = SpeculativeDecodingMode.from_arguments(
            args)

        if args.build_config is None:
            if args.multiple_profiles == "enable" and args.opt_num_tokens is not None:
                raise RuntimeError(
                    "multiple_profiles is enabled, while opt_num_tokens is set. "
                    "They are not supposed to be working in the same time for now."
                )
            if args.cluster_key is not None:
                cluster_config = dict(cluster_key=args.cluster_key)
            else:
                cluster_config = infer_cluster_config()

            # This should only be used for debugging.
            # The env var BUILDER_FORCE_NUM_PROFILES should override the number of
            # optimization profiles during TRT build.
            # BUILDER_FORCE_NUM_PROFILES must be less than or equal to the number of
            # optimization profiles set by model's prepare_inputs().
            force_num_profiles_from_env = os.environ.get(
                "BUILDER_FORCE_NUM_PROFILES", None)
            if force_num_profiles_from_env is not None:
                logger.warning(
                    f"Overriding # of builder profiles <= {force_num_profiles_from_env}."
                )

            build_config = BuildConfig.from_dict(
                {
                    'max_input_len': args.max_input_len,
                    'max_seq_len': args.max_seq_len,
                    'max_batch_size': args.max_batch_size,
                    'max_beam_width': args.max_beam_width,
                    'max_num_tokens': args.max_num_tokens,
                    'opt_num_tokens': args.opt_num_tokens,
                    'max_prompt_embedding_table_size':
                    args.max_prompt_embedding_table_size,
                    'gather_context_logits': args.gather_context_logits,
                    'gather_generation_logits': args.gather_generation_logits,
                    'strongly_typed': True,
                    'force_num_profiles': force_num_profiles_from_env,
                    'weight_sparsity': args.weight_sparsity,
                    'profiling_verbosity': args.profiling_verbosity,
                    'enable_debug_output': args.enable_debug_output,
                    'max_draft_len': args.max_draft_len,
                    'speculative_decoding_mode': speculative_decoding_mode,
                    'input_timing_cache': args.input_timing_cache,
                    'output_timing_cache': '/tmp/model.cache',
                    'auto_parallel_config': {
                        'world_size':
                        args.auto_parallel,
                        'gpus_per_node':
                        args.gpus_per_node,
                        'sharded_io_allowlist': [
                            'past_key_value_\\d+',
                            'present_key_value_\\d*',
                        ],
                        'same_buffer_io': {
                            'past_key_value_(\\d+)': 'present_key_value_\\1',
                        },
                        **cluster_config,
                    },
                    'dry_run': args.dry_run,
                    'visualize_network': args.visualize_network,
                    'max_encoder_input_len': args.max_encoder_input_len,
                    'weight_streaming': args.weight_streaming,
                    'monitor_memory': args.monitor_memory,
                },
                plugin_config=plugin_config)

            if hasattr(args, 'kv_cache_type'):
                build_config.update_from_dict(
                    {'kv_cache_type': args.kv_cache_type})
        else:
            build_config = BuildConfig.from_json_file(
                args.build_config, plugin_config=plugin_config)
        return build_config

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
            "moe_tensor_parallel_size": self.moe_tensor_parallel_size,
            "moe_expert_parallel_size": self.moe_expert_parallel_size,
            "enable_attention_dp": self.enable_attention_dp,
            "auto_parallel": self.auto_parallel,
            "auto_parallel_world_size": self.auto_parallel_world_size,
            "load_format": self.load_format,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "guided_decoding_backend": self.guided_decoding_backend,
            "iter_stats_max_iterations": self.iter_stats_max_iterations,
            "request_stats_max_iterations": self.request_stats_max_iterations,
            "embedding_parallel_mode": self.embedding_parallel_mode,
            "fast_build": self.fast_build,
            "enable_build_cache": self.enable_build_cache,
            "batching_type": self.batching_type,
            "normalize_log_probs": self.normalize_log_probs,
            "gather_generation_logits": self.gather_generation_logits,
            "extended_runtime_perf_knob_config":
            self.extended_runtime_perf_knob_config,
            "max_batch_size": self.max_rolling_batch_size,
            "max_input_len": self.max_input_len,
            "max_seq_len": self.max_seq_len,
            "max_beam_width": self.max_beam_width,
            "max_num_tokens": self.max_num_tokens,
            "backend": self.backend,
            "quant_config": self.get_quant_config(),
            "calib_config": self.get_calib_config(),
            "build_config": self.get_build_config(),
            "kv_cache_config": self.get_kv_cache_config(),
        }
