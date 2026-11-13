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
        d_x = out_grad * y.data
        d_y = out_grad * x.data
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
        d_x = out_grad / y.data
        d_y = (-out_grad * x.data) / (np.power(y.data, 2))
        return d_x, d_y


class Pow(Function):
    def forward(self, x, y):
        return np.power(x, y)

    def backward(self, out_grad, node):
        x, y = node._inputs
        d_x = out_grad * y.data * np.power(x.data, y.data - 1)
        d_y = out_grad * np.power(x.data, y.data) * np.log(x.data)
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


class MatMul(Function):
    def forward(self, x, y):
        return np.matmul(x, y)

    def backward(self, out_grad, node):
        x, y = node._inputs
        d_x = np.matmul(out_grad, y.data.T)
        d_y = np.matmul(x.data.T, out_grad)
        return d_x, d_y


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return np.reshape(x, shape=self.shape)
        
    def backward(self, out_grad, node):
        x = node._inputs[0]
        d_x = np.reshape(out_grad, x.shape)
        return d_x


class Summation(Function):
    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def forward(self, x):
        return np.sum(x, axis=self.axes)

    def backward(self, out_grad, node):
        # la derivata della sommatoria e' uguale alla sommatoria delle derivate dei singoli termini
        x = node._inputs[0]

        # global sum
        if self.axes is None:
            out_grad = np.expand_dims(out_grad, axis=0) # qui l'asse e' indifferente
            out_grad = np.broadcast_to(out_grad, x.shape)
        # specific axes
        else:
            out_grad = np.expand_dims(out_grad, axis=self.axes)
            out_grad = np.broadcast_to(out_grad, x.shape)

        return out_grad


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return np.broadcast_to(x, shape=self.shape)

    def backward(self, out_grad, node):
        x = node._inputs[0]
        ndims_added = len(self.shape) - len(x.shape)
        padded_shape = (1,) * ndims_added + x.shape
        axes_to_sum = tuple(
            i for i, (orig_dim, out_dim) in enumerate(zip(padded_shape, self.shape))
            if orig_dim == 1 and out_dim != 1
        )
        d_x = np.sum(out_grad, axis=axes_to_sum)
        return np.reshape(d_x, x.shape)


class ReLU(Function):
    def forward(self, x):
        return np.max(0, x)

    def backward(self, out_grad, node):
        x = node._inputs[0]
        d_x = Tensor((x.data > 0).astype("float32"), requires_grad=False)
        return np.multiply(out_grad, d_x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, out_grad, node):
        x = node._inputs[0]
        d_x = 1 - np.tanh(x) ** 2
        return d_x


class Sigmoid(Function):
    def forward(self, x):
        return 1 / (1 + np.power(np.exp(1), -x))

    def backward(self, out_grad, node):
        d_x = out_grad * (1 - out_grad)
        return d_x


def sigmoid(x):
    return Sigmoid()(x)


def tanh(x):
    return Tanh()(x)


def relu(x):
    return ReLU()(x)


def summation(x, axes = None):
    return Summation(axes)(x)


def reshape(x, shape = None):
    return Reshape(shape)(x)


def matmul(x, y):
    return MatMul()(x, y)


def transpose(x, axes = None):
    return Transpose(axes)(x)


def add(x, y):
    return Add()(x, y)


def mul(x, y):
    return Mul()(x, y)


def sub(x, y):
    return Sub()(x, y)


def div(x, y):
    return Div()(x, y)


def pow(x, y):
    return Pow()(x, y)
