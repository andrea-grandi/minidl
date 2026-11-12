import numpy as np
from minidl.tensor import Tensor
from minidl.ops import Transpose

a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])

c = a + b
print("add:", c)             # [5. 7. 9.]
print("op:", c._op)          # <minidl.ops.Add ...>

d = a * b
print("mul:", d)             # [ 4. 10. 18.]
print("op:", d._op)

x = Tensor(np.arange(24).reshape(2, 3, 4))
op = Transpose((1, 2, 0))
y = op(x)
print("transpose shape:", x.shape, "->", y.shape)   # (2,3,4) -> (3,4,2)

# backward a mano
g = np.ones(y.shape, dtype="float32")
print("grad shape:", op.backward(g, y).shape)       # deve tornare (2,3,4)
