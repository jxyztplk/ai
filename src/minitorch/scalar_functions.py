from __future__ import annotations

from typing import TYPE_CHECKING

from minitorch import scalar

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = scalar.ScalarHistory(cls, ctx, scalars)
        return scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)



# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return d_output, d_output


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -1 / pow(a, 2) * d_output


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # Compute the forward pass: -a
        ctx.save_for_backward(a)  # Save input for backward pass
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # Retrieve input from context
        a, = ctx.saved_tensors
        # Compute the backward pass: -d_output
        return -d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # Compute the forward pass: sigmoid(a) = 1 / (1 + exp(-a))
        sig_a = 1 / (1 + exp(-a))
        ctx.save_for_backward(sig_a)  # Save output for backward pass
        return sig_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # Retrieve output from context
        sig_a, = ctx.saved_tensors
        # Compute the backward pass: sigmoid'(a) = sigmoid(a) * (1 - sigmoid(a))
        return d_output * sig_a * (1 - sig_a)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # Compute the forward pass: max(0, a)
        ctx.save_for_backward(a)  # Save input for backward pass
        return max(0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # Retrieve input from context
        a, = ctx.saved_tensors
        # Compute the backward pass: ReLU'(a) = 1 if a > 0 else 0
        return d_output if a > 0 else 0


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # Compute the forward pass: exp(a)
        exp_a = exp(a)
        ctx.save_for_backward(exp_a)  # Save output for backward pass
        return exp_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # Retrieve output from context
        exp_a, = ctx.saved_tensors
        # Compute the backward pass: exp'(a) = exp(a)
        return d_output * exp_a


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # Compute the forward pass: 1.0 if a < b else 0.0
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # Derivative with respect to a and b is 0 since they are constants
        return 0.0, 0.0


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # Compute the forward pass: 1.0 if a == b else 0.0
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # Derivative with respect to a and b is 0 since they are constants
        return 0.0, 0.0
