import os
from typing import Optional

from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.lmi_inference_service import LmiInferenceService

_service: Optional[LmiInferenceService] = None


def determine_lmi_inference_service(
        properties: dict, serving_features: str) -> LmiInferenceService:
    rolling_batch = properties.get("rolling_batch", "disable")
    if "tnx" in serving_features:
        if properties.get("use_stable_diffusion", False):
            from .stable_diffusion_inf2 import StableDiffusionNeuronXService
            return StableDiffusionNeuronXService()
        from .transformers_neuronx import TransformersNeuronXService
        return TransformersNeuronXService()
    if "trtllm" in serving_features:
        if rolling_batch == "disable":
            from .tensorrt_llm_python import TRTLLMPythonService
            return TRTLLMPythonService()
        from .tensorrt_llm import TRTLLMService
        return TRTLLMService()
    else:
        if rolling_batch == "vllm":
            from .lmi_inference_service.vllm_inference_service import VllmInferenceService
            return VllmInferenceService()
        from .huggingface import HuggingFaceService
        return HuggingFaceService()


def handle(inputs: Input) -> Optional[Output]:
    global _service
    if _service is None or not _service.is_initialized():
        properties = inputs.get_properties()
        serving_features = os.environ.get("SERVING_FEATURES")
        _service = determine_lmi_inference_service(properties,
                                                   serving_features)
        _service.initialize(properties)
    if inputs.is_empty():
        return None
    return _service.inference(inputs)
