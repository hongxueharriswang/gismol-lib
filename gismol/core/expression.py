# ============================================================
# FILE: gismol/core/expression.py
# ============================================================
"""Symbolic expression language used to express COH constraints.

Expressions are immutable trees that evaluate against a *state* (any mapping,
including a hierarchical :class:`~gismol.core.coh_object.State`)::

    x = Var("x")
    expr = (x > 0) & (x < 10)
    expr.evaluate({"x": 3})        # -> True

They can also be parsed from a side-effect free subset of Python syntax::

    parse_expression("x > 0 and x < 10")

Parsing never calls :func:`eval`; only :mod:`ast` nodes on a whitelist are
accepted.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Dict, List, Mapping, Set

from .exceptions import ParseError

__all__ = [
    "Expr", "Const", "Var", "Cmp", "Arith", "Call", "And", "Or", "Not", "Implies",
    "lookup", "register_function", "parse_expression", "evaluate",
]


# --------------------------------------------------------------------------
# function registry (used by Call nodes)
# --------------------------------------------------------------------------
_FUNCTIONS: Dict[str, Callable[..., Any]] = {}


def register_function(name: str, fn: Callable[..., Any] | None = None):
    """Register *fn* under *name* so it can be used inside parsed expressions."""
    if fn is None:
        def decorator(f):
            _FUNCTIONS[name] = f
            return f
        return decorator
    _FUNCTIONS[name] = fn
    return fn


def _builtin_functions() -> Dict[str, Callable[..., Any]]:
    return {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "round": round, "int": int, "float": float, "bool": bool,
        "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
        "sin": math.sin, "cos": math.cos, "tanh": math.tanh,
        "true": lambda: True, "false": lambda: False,
    }


_FUNCTIONS.update(_builtin_functions())


# --------------------------------------------------------------------------
# state lookup helper
# --------------------------------------------------------------------------
_MISSING = object()


def lookup(state: Any, name: str, default: Any = _MISSING) -> Any:
    """Resolve *name* against *state*.

    Supports plain mappings, dotted paths (``"arm.joint.angle"``) and objects
    exposing ``get_path`` (i.e. :class:`State`).
    """
    try:
        if name in state:                       # mapping fast-path
            return state[name]
    except TypeError:
        pass
    getter = getattr(state, "get_path", None)
    if callable(getter):
        value = getter(name, _MISSING)
        if value is not _MISSING:
            return value
    if default is not _MISSING:
        return default
    raise KeyError(f"unknown state variable {name!r}")


# --------------------------------------------------------------------------
# expression nodes
# --------------------------------------------------------------------------
class Expr:
    """Base class of the COH expression language."""

    #: arity of children, used by ``children()``
    def children(self) -> List["Expr"]:
        return []

    def evaluate(self, state: Any) -> Any:
        raise NotImplementedError

    def variables(self) -> Set[str]:
        out: Set[str] = set()
        for child in self.children():
            out |= child.variables()
        return out

    # -- boolean algebra ---------------------------------------------------
    def __and__(self, other):  return And(self, as_expr(other))
    def __or__(self, other):   return Or(self, as_expr(other))
    def __invert__(self):      return Not(self)
    def __rand__(self, other): return And(as_expr(other), self)
    def __ror__(self, other):  return Or(as_expr(other), self)

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other):      return Arith("+", self, as_expr(other))
    def __radd__(self, other):     return Arith("+", as_expr(other), self)
    def __sub__(self, other):      return Arith("-", self, as_expr(other))
    def __rsub__(self, other):     return Arith("-", as_expr(other), self)
    def __mul__(self, other):      return Arith("*", self, as_expr(other))
    def __rmul__(self, other):     return Arith("*", as_expr(other), self)
    def __truediv__(self, other):  return Arith("/", self, as_expr(other))
    def __rtruediv__(self, other): return Arith("/", as_expr(other), self)
    def __mod__(self, other):      return Arith("%", self, as_expr(other))
    def __pow__(self, other):      return Arith("**", self, as_expr(other))
    def __neg__(self):             return Arith("neg", self)

    # -- comparisons -------------------------------------------------------
    def __gt__(self, other): return Cmp(">", self, as_expr(other))
    def __ge__(self, other): return Cmp(">=", self, as_expr(other))
    def __lt__(self, other): return Cmp("<", self, as_expr(other))
    def __le__(self, other): return Cmp("<=", self, as_expr(other))

    def eq(self, other):  return Cmp("==", self, as_expr(other))
    def ne(self, other):  return Cmp("!=", self, as_expr(other))
    def contains(self, other): return Cmp("in", as_expr(other), self)

    def __bool__(self):  # pragma: no cover - defensive
        raise TypeError(
            "COH expressions cannot be used as Python booleans; "
            "use '&', '|' and '~' instead of 'and', 'or' and 'not'."
        )

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"op": type(self).__name__.lower(), "args": [c.to_dict() for c in self.children()]}

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class Const(Expr):
    def __init__(self, value: Any):
        self.value = value

    def evaluate(self, state): return self.value

    def to_dict(self): return {"op": "const", "value": self.value}

    def __repr__(self): return repr(self.value)


class Var(Expr):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, state): return lookup(state, self.name)

    def variables(self): return {self.name}

    def to_dict(self): return {"op": "var", "name": self.name}

    def __repr__(self): return self.name


class _Nary(Expr):
    symbol = "?"

    def __init__(self, *operands):
        self.operands = [as_expr(o) for o in operands]

    def children(self): return list(self.operands)

    def to_dict(self):
        return {"op": type(self).__name__.lower(),
                "args": [o.to_dict() for o in self.operands]}

    def __repr__(self):
        return "(" + f" {self.symbol} ".join(map(repr, self.operands)) + ")"


class And(_Nary):
    symbol = "and"

    def evaluate(self, state):
        return all(bool(o.evaluate(state)) for o in self.operands)


class Or(_Nary):
    symbol = "or"

    def evaluate(self, state):
        return any(bool(o.evaluate(state)) for o in self.operands)


class Not(Expr):
    def __init__(self, operand):
        self.operand = as_expr(operand)

    def children(self): return [self.operand]

    def evaluate(self, state): return not bool(self.operand.evaluate(state))

    def to_dict(self): return {"op": "not", "arg": self.operand.to_dict()}

    def __repr__(self): return f"(not {self.operand!r})"


class Implies(Expr):
    def __init__(self, antecedent, consequent):
        self.antecedent = as_expr(antecedent)
        self.consequent = as_expr(consequent)

    def children(self): return [self.antecedent, self.consequent]

    def evaluate(self, state):
        return (not bool(self.antecedent.evaluate(state))) or bool(
            self.consequent.evaluate(state)
        )

    def to_dict(self):
        return {"op": "implies", "args": [self.antecedent.to_dict(),
                                          self.consequent.to_dict()]}

    def __repr__(self): return f"({self.antecedent!r} -> {self.consequent!r})"


_CMP_OPS = {
    "==": operator.eq, "!=": operator.ne,
    "<": operator.lt, "<=": operator.le,
    ">": operator.gt, ">=": operator.ge,
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
    "is": operator.is_,
    "is not": operator.is_not,
}


class Cmp(Expr):
    def __init__(self, op: str, left, right):
        if op not in _CMP_OPS:
            raise ParseError(f"unsupported comparison operator {op!r}")
        self.op = op
        self.left = as_expr(left)
        self.right = as_expr(right)

    def children(self): return [self.left, self.right]

    def evaluate(self, state):
        a = self.left.evaluate(state)
        b = self.right.evaluate(state)
        try:
            return bool(_CMP_OPS[self.op](a, b))
        except TypeError:
            return False

    def to_dict(self):
        return {"op": "cmp", "operator": self.op,
                "args": [self.left.to_dict(), self.right.to_dict()]}

    def __repr__(self): return f"({self.left!r} {self.op} {self.right!r})"


_ARITH_OPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": operator.truediv, "%": operator.mod, "**": operator.pow,
}


class Arith(Expr):
    def __init__(self, op: str, left, right=None):
        self.op = op
        self.left = as_expr(left)
        self.right = None if right is None else as_expr(right)

    def children(self):
        return [self.left] if self.right is None else [self.left, self.right]

    def evaluate(self, state):
        a = self.left.evaluate(state)
        if self.op == "neg":
            return -a
        b = self.right.evaluate(state)
        return _ARITH_OPS[self.op](a, b)

    def to_dict(self):
        args = [self.left.to_dict()] + ([self.right.to_dict()] if self.right else [])
        return {"op": "arith", "operator": self.op, "args": args}

    def __repr__(self):
        if self.right is None:
            return f"(-{self.left!r})"
        return f"({self.left!r} {self.op} {self.right!r})"


class Call(Expr):
    """Named function application, resolved through the global registry."""

    def __init__(self, name: str, args=()):
        self.name = name
        self.args = [as_expr(a) for a in args]

    def children(self): return list(self.args)

    def evaluate(self, state):
        fn = _FUNCTIONS.get(self.name)
        if fn is None:
            raise ParseError(f"unknown function {self.name!r}")
        return fn(*[a.evaluate(state) for a in self.args])

    def to_dict(self):
        return {"op": "call", "name": self.name,
                "args": [a.to_dict() for a in self.args]}

    def __repr__(self): return f"{self.name}({', '.join(map(repr, self.args))})"


def as_expr(value) -> Expr:
    """Coerce *value* into an :class:`Expr`."""
    if isinstance(value, Expr):
        return value
    if callable(value):
        return Call(getattr(value, "__name__", "fn"), ())
    return Const(value)


# --------------------------------------------------------------------------
# evaluation helper
# --------------------------------------------------------------------------
def evaluate(expr, state) -> Any:
    """Evaluate *expr* (an :class:`Expr` or a plain callable) against *state*."""
    if isinstance(expr, Expr):
        return expr.evaluate(state)
    if callable(expr):
        return expr(state)
    return expr


# --------------------------------------------------------------------------
# parser (safe subset of Python expression syntax)
# --------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.Mod: "%", ast.Pow: "**",
}
_CMP_OPS_AST = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=", ast.In: "in", ast.NotIn: "not in",
    ast.Is: "is", ast.IsNot: "is not",
}


def parse_expression(source: str) -> Expr:
    """Parse *source* into an :class:`Expr` tree.

    Supports literals, names, comparisons, arithmetic, boolean operators,
    parentheses and calls to registered functions.  ``if``/``lambda``/
    comprehensions/attribute access/subscription are rejected.
    """
    if isinstance(source, Expr):
        return source
    text = str(source).strip()
    if not text:
        raise ParseError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ParseError(f"cannot parse expression {source!r}: {exc}") from exc
    return _from_ast(tree.body)


def _from_ast(node: ast.AST) -> Expr:
    if isinstance(node, ast.Constant):
        return Const(node.value)

    if isinstance(node, ast.Name):
        if node.id == "True":
            return Const(True)
        if node.id == "False":
            return Const(False)
        if node.id == "None":
            return Const(None)
        return Var(node.id)

    if isinstance(node, ast.BoolOp):
        operands = [_from_ast(v) for v in node.values]
        cls = And if isinstance(node.op, ast.And) else Or
        return cls(*operands)

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return Not(_from_ast(node.operand))
        if isinstance(node.op, ast.USub):
            return Arith("neg", _from_ast(node.operand))
        if isinstance(node.op, ast.UAdd):
            return _from_ast(node.operand)
        raise ParseError(f"unsupported unary operator {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ParseError(f"unsupported binary operator {type(node.op).__name__}")
        return Arith(op, _from_ast(node.left), _from_ast(node.right))

    if isinstance(node, ast.Compare):
        parts: List[Expr] = []
        left = _from_ast(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMP_OPS_AST.get(type(op_node))
            if op is None:
                raise ParseError(f"unsupported comparison {type(op_node).__name__}")
            right = _from_ast(comparator)
            parts.append(Cmp(op, left, right))
            left = right
        return parts[0] if len(parts) == 1 else And(*parts)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ParseError("only simple function calls are allowed")
        if node.keywords:
            raise ParseError("keyword arguments are not supported")
        return Call(node.func.id, [_from_ast(a) for a in node.args])

    if isinstance(node, (ast.Tuple, ast.List)):
        return Const(tuple(_from_ast(e) for e in node.elts))

    raise ParseError(f"unsupported expression element {type(node).__name__}")
