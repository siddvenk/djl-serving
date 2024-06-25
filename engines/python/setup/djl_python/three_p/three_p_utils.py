from typing import Optional


def is_3p_request(invoke_type: Optional[str]) -> bool:
    # TODO, not sure if this is reliable
    # We might want to just use an env var since in the 3p env will will only run in 1 way
    return invoke_type == "InvokeEndpoint" or invoke_type == "InvokeEndpointWithResponseStream"


def parse_3p_request(input_map: dict, is_rolling_batch: bool, tokenizer,
                     invoke_type: str):
    _inputs = input_map.pop("prompt")
    _param = {"details": True}
    _param["temperature"] = input_map.pop("temperature", 0.5)
    _param["top_p"] = input_map.pop("top_p", 0.9)
    _param["max_new_tokens"] = input_map.pop("max_gen_len", 512)
    if invoke_type == "InvokeEndpointWithResponseStream":
        _param["stream"] = True
        _param["output_formatter"] = "3p_stream"
    else:
        _param["output_formatter"] = "3p"
    return _inputs, _param
