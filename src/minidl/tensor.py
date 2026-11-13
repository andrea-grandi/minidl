import numpy as np
from typing import Any


class Tensor:
    def __init__(self, data: Any, *, device: Any | None = None, dtype: str = "float32", requires_grad: bool = True):
        """
        Create a new tensor.
        Args:
            data: Array-like data (list, numpy array, or another Tensor)
            device: Device placement (currently ignored, CPU only)
            dtype: Data type for the array
            requires_grad: Whether to track gradients for this tensor
        """
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            self.data = data.numpy().astype(dtype)

        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype if dtype is not None else data.dtype)

        else:
            self.data = np.array(data, dtype if dtype is not None else "float32")

        self.requires_grad = requires_grad
        self.grad = None
        self._op = None
        self._inputs = []
        self._device = device if device else "cpu"

    def __repr__(self):
        return f"Tensor ({self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        return str(self.data)

    def __add__(self, other):
        from .ops import Add
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add()(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        from .ops import Mul
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Mul()(self, other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def device(self):
        return self._device

    @property
    def T(self):
        return np.transpose(self)

    def backward(self, out_grad):
        pass
    
    def numpy(self):
        return self.data.copy()

    def detach(self):
        return Tensor(self.data, dtype=self.dtype, requires_grad=False)
