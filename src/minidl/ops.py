import numpy as np

from .tensor import Tensor


class Function:
    def __call__(self, *inputs):
        """
        1. Takes the input Tensor
        2. Gets .data from those inputs
        3. Calls forward() using data from inputs
        4. Wraps the output to a Tensor
        5. Return
        """
        requires_grad = any(t.requires_grad for t in inputs)
        input_data = [t.data for t in inputs]
        output_data = self.forward(*input_data)
        output_tensor = Tensor(data=output_data, dtype=input_data[0].dtype, requires_grad=requires_grad)
        
        if requires_grad:
            output_tensor._op = self
            output_tensor._inputs = inputs
        
        return output_tensor
    
    def forward(self, *args):
        raise NotImplementedError()

    def backward(self, out_grad, node):
        pass


class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, out_grad, node):
        d_x = out_grad
        d_y = out_grad
        return d_x, d_y


class Mul(Function):
    def forward(self, x, y):
        return x * y
    
    def backward(self, out_grad, node):
        x, y = node._inputs
        d_x = out_grad * y
        d_y = out_grad * x
        return d_x, d_y


class Sub(Function):
    def forward(self, x, y):
        return x - y

    def backward(self, out_grad, node):
        d_x = out_grad
        d_y = -out_grad
        return d_x, d_y


class Div(Function):
    def forward(self, x, y):
        return x / y

    def backward(self, out_grad, node):
        x, y = node._inputs
        d_x = out_grad / y
        d_y = (-out_grad * x) / (np.power(y, 2))
        return d_x, d_y


class Pow(Function):
    def forward(self, x, y):
        return np.power(x, y)

    def backward(self, out_grad, node):
        x, y = node._inputs
        d_x = out_grad * y * np.power(x, y - 1)
        d_y = out_grad * np.power(x, y) * np.log(x)
        return d_x, d_y


class Transpose(Function):
    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def forward(self, x):
        return np.transpose(x, self.axes)
        
    def backward(self, out_grad, node):
        if self.axes is None:
            return np.transpose(out_grad)
        inverse_axes = np.argsort(self.axes)
        return np.transpose(out_grad, tuple(inverse_axes))


def add(x, y):
    return Add()(x, y)


def mul(x, y):
    return Mul()(x, y)


def sub(x, y):
    return Sub()(x, y)


def div(x, y):
    return Div()(x, y)


def Paw(x, y):
    return Pow()(x, y)
