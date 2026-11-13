import numpy as np

from .tensor import Tensor
from .ops import relu


class Parameter(Tensor):
    def __init__(self, data, *args, **kwargs):
        kwargs["requires_grad"] = True
        super().__init__(data, *args, **kwargs)


class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        params = _get_parameters(self.__dict__)
        unique_params = []
        seen_ids = set()
        for p in params:
            if id(p) not in seen_ids:
                unique_params.append(p)
                seen_ids.add(id(p))
        return unique_params

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True 
        for m in _get_modules(self.__dict__):
            m.training = True 
    def eval(self):
        self.training = False 
        for m in _get_modules(self.__dict__):
            m.training = False 


class ReLU(Module):
    def forward(self, x):
        return relu(x)


def _get_modules(obj) -> list['Module']:
    """
    A simple recursive function that finds all Module objects within any given
    object by searching through its attributes, lists, tuples, and dicts.
    """
    modules = []
    if isinstance(obj, Module):
        return [obj]
        
    if isinstance(obj, dict):
        for value in obj.values():
            modules.extend(_get_modules(value))

    if isinstance(obj, (list, tuple)):
        for item in obj:
            modules.extend(_get_modules(item))

    return modules


def _get_parameters(data):
    params = []
    if isinstance(data, Parameter):
        return [data]
    if isinstance(data, dict):
        for value in data.values(): #calling _get_parameters recursively
            params.extend(_get_parameters(value))
    if isinstance(data, (list, tuple)):
        for item in data:
            params.extend(_get_parameters(item))
    return params
