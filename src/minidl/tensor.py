import numpy as np
from typing import Any


def _ensure_tensor(val):
    return val if isinstance(val, Tensor) else Tensor(val,
     requires_grad=False)


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
        return self.transpose()   
    
    @classmethod
    def rand(cls, *shape, low=0.0, high=1.0, dtype="float32", 
        requires_grad=True):
        array = np.random.rand(*shape) * (high - low) + low
        return cls(array.astype(dtype), requires_grad=requires_grad)
    
    @classmethod
    def randn(cls, *shape, mean=0.0, std=1.0, dtype="float32",
         requires_grad=True):
        array = np.random.randn(*shape) * std + mean
        return cls(array.astype(dtype), requires_grad=requires_grad)
    
    @classmethod
    def constant(cls, *shape, c=1.0, dtype="float32", requires_grad=True):
        array = np.ones(*shape) * c
        return cls(array.astype(dtype), requires_grad=requires_grad)
    
    @classmethod
    def ones(cls, *shape, dtype="float32", requires_grad=True):
        return cls.constant(*shape, c=1.0, dtype=dtype, 
        requires_grad=requires_grad)
    
    @classmethod
    def zeros(cls, *shape, dtype="float32", requires_grad=True):
        return cls.constant(*shape, c=0.0, dtype=dtype,
         requires_grad=requires_grad)
    
    @classmethod 
    def randb(cls, *shape, p=0.5, dtype="float32", requires_grad=True):
        array =np.random.rand(*shape) <=p 
        return cls(array,dtype=dtype, requires_grad=requires_grad )
    
    @classmethod 
    def empty(cls, *shape, dtype="float32", requires_grad=True):
        array =np.empty(shape,dtype=dtype)
        return cls(array, requires_grad=requires_grad)
    
    @classmethod 
    def one_hot(cls,indices,num_classes,device=None, dtype="float32",
         requires_grad=True):
        one_hot_array = np.eye(num_classes,dtype=dtype)[np.array(
            indices.data,dtype=int)]
        return cls(one_hot_array,device=device, dtype=dtype,
            requires_grad=requires_grad)

    def numpy(self):
            return self.data.copy()
    
    def detach(self):
        return Tensor(self.data, dtype=self.dtype, requires_grad=False)

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")

        # Build the "Family Tree" in order (Topological Sort)
        topo_order = []
        visited = set()
        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._inputs:
                    build_topo(parent)
                topo_order.append(node)
        build_topo(self)
        
        # Initialize the Ledger
        grads = {}
        if grad is None:
            # The "output" gradient: dL/dL = 1
            grads[id(self)] = Tensor(np.ones_like(self.data))
        else:
            grads[id(self)] = _ensure_tensor(grad)
        
        # Walk the Graph Backwards
        for node in reversed(topo_order):
            out_grad = grads.get(id(node))
            if out_grad is None:
                continue

            # Store the final result in the .grad attribute
            if node.grad is None:
                node.grad = np.array(out_grad.data, copy=True)

            else:
                node.grad += out_grad.data

            # Propagate to Parents
            if node._op:
                input_grads = node._op.backward(out_grad, node)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)
                
                for i, parent in enumerate(node._inputs):
                    if parent.requires_grad:
                        parent_id = id(parent)
                        if parent_id not in grads:
                            # First time seeing this parent
                            grads[parent_id] = input_grads[i]
                        else:
                            #  Sum the gradients!
                            grads[parent_id] = grads[parent_id] + input_grads[i]
