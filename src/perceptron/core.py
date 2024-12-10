import numpy as np


# Custom Tensor class
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        # self.grad = None if not requires_grad else np.zeros_like(self.data)

    # def backward(self, grad=None):
    #     if not self.requires_grad:
    #         raise RuntimeError("Called backward on a tensor that does not require gradients.")
    #     if grad is None:
    #         grad = np.ones_like(self.data)
    #     self.grad += grad
    #     if hasattr(self, 'grad_fn') and self.grad_fn is not None:
    #         self.grad_fn(grad)

    def __add__(self, other):
        other = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data + other)
        # if self.requires_grad:
        #   result.grad_fn = lambda grad: self.backward(grad)
        return result

    def __matmul__(self, other):
        other = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data @ other)
        # if self.requires_grad:
        #   result.grad_fn = lambda grad: self.backward(grad @ other.T)
        return result

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"


class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=False)
        self.bias = Tensor(np.zeros(out_features), requires_grad=False)

    def __call__(self, x):
        return x @ self.weights + self.bias
