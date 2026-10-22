"""Explicit older nonnumerical support; not a full DiffusionTrainingModule."""
import torch


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def fetch_input_params(self):
        params = []
        if self.input_params is not None:
            for param in self.input_params:
                params.append(param)
        if self.input_params_posi is not None:
            for _, param in self.input_params_posi.items():
                params.append(param)
        if self.input_params_nega is not None:
            for _, param in self.input_params_nega.items():
                params.append(param)
        params = sorted(list(set(params)))
        return params
    
    def fetch_output_params(self):
        params = []
        if self.output_params is not None:
            for param in self.output_params:
                params.append(param)
        return params

    def process(self, pipe, **kwargs) -> dict:
        return {}
    
    def post_process(self, pipe, **kwargs) -> dict:
        return {}


class CachedTensorMover:
    """Maintained limited wrapper around the actual retained tensor mover."""

    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data
