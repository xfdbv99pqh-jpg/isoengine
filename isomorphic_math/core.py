"""
Core components: Expression system, Parser, Solvers
"""

import math
import re
from enum import Enum
from typing import List, Tuple, Optional, Dict, Union
import random

# Check for torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# EXPRESSION SYSTEM
# ============================================================================

class Op(Enum):
    """All supported operators."""
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    POW = 4
    NEG = 5
    VAR_X = 6
    VAR_Y = 7
    VAR_Z = 8
    CONST = 9
    EQ = 10
    GT = 11
    LT = 12
    GTE = 13
    LTE = 14
    NEQ = 15
    SYSTEM = 16
    OR = 17
    AND = 18
    DERIV = 19
    INTEGRAL = 20
    SIN = 21
    COS = 22
    TAN = 23
    LN = 24
    EXP = 25
    SQRT = 26
    ABS = 27
    PAD = 28


class ProblemType(Enum):
    """Problem classification."""
    LINEAR_EQ = 0
    QUADRATIC_EQ = 1
    SYSTEM_EQ = 2
    LINEAR_INEQ = 3
    QUADRATIC_INEQ = 4
    DERIVATIVE = 5
    INTEGRAL = 6
    TRIG_EQ = 7
    INVALID = 8
    UNKNOWN = 9


class Expr:
    """Expression tree node."""
    
    def __init__(self, op: Op, value: float = 0.0,
                 left: 'Expr' = None, right: 'Expr' = None):
        self.op = op
        self.value = value
        self.left = left
        self.right = right
    
    def __repr__(self):
        if self.op == Op.CONST:
            v = self.value
            return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.4g}"
        elif self.op == Op.VAR_X: return "x"
        elif self.op == Op.VAR_Y: return "y"
        elif self.op == Op.VAR_Z: return "z"
        elif self.op == Op.ADD: return f"({self.left} + {self.right})"
        elif self.op == Op.SUB: return f"({self.left} - {self.right})"
        elif self.op == Op.MUL: return f"({self.left} * {self.right})"
        elif self.op == Op.DIV: return f"({self.left} / {self.right})"
        elif self.op == Op.POW: return f"({self.left}^{self.right})"
        elif self.op == Op.NEG: return f"(-{self.left})"
        elif self.op == Op.EQ: return f"{self.left} = {self.right}"
        elif self.op == Op.GT: return f"{self.left} > {self.right}"
        elif self.op == Op.LT: return f"{self.left} < {self.right}"
        elif self.op == Op.GTE: return f"{self.left} >= {self.right}"
        elif self.op == Op.LTE: return f"{self.left} <= {self.right}"
        elif self.op == Op.SYSTEM: return f"{{ {self.left} ; {self.right} }}"
        elif self.op == Op.OR: return f"{self.left} OR {self.right}"
        elif self.op == Op.AND: return f"{self.left} AND {self.right}"
        elif self.op == Op.DERIV: return f"d/dx[{self.left}]"
        elif self.op == Op.SIN: return f"sin({self.left})"
        elif self.op == Op.COS: return f"cos({self.left})"
        elif self.op == Op.TAN: return f"tan({self.left})"
        elif self.op == Op.LN: return f"ln({self.left})"
        elif self.op == Op.EXP: return f"e^({self.left})"
        elif self.op == Op.SQRT: return f"√({self.left})"
        elif self.op == Op.ABS: return f"|{self.left}|"
        return "?"
    
    def copy(self) -> 'Expr':
        return Expr(self.op, self.value,
                   self.left.copy() if self.left else None,
                   self.right.copy() if self.right else None)
    
    def equals(self, other: 'Expr') -> bool:
        if other is None or self.op != other.op:
            return False
        if self.op == Op.CONST:
            return abs(self.value - other.value) < 1e-9
        if self.op in [Op.VAR_X, Op.VAR_Y, Op.VAR_Z, Op.PAD]:
            return True
        left_eq = (self.left is None and other.left is None) or \
                  (self.left and self.left.equals(other.left))
        right_eq = (self.right is None and other.right is None) or \
                   (self.right and self.right.equals(other.right))
        return left_eq and right_eq
    
    def to_tensor(self, max_nodes: int = 32):
        """Convert to tensor representation."""
        ops, vals = [], []
        def traverse(node):
            if node is None or len(ops) >= max_nodes:
                return
            ops.append(node.op.value)
            vals.append(node.value if node.op == Op.CONST else 0.0)
            traverse(node.left)
            traverse(node.right)
        traverse(self)
        while len(ops) < max_nodes:
            ops.append(Op.PAD.value)
            vals.append(0.0)
        if TORCH_AVAILABLE:
            return (torch.tensor(ops[:max_nodes]),
                    torch.tensor(vals[:max_nodes], dtype=torch.float32))
        return ops[:max_nodes], vals[:max_nodes]
    
    def evaluate(self, x_val: float = 0, y_val: float = 0) -> float:
        """Numerically evaluate."""
        if self.op == Op.CONST: return self.value
        elif self.op == Op.VAR_X: return x_val
        elif self.op == Op.VAR_Y: return y_val
        elif self.op == Op.ADD: return self.left.evaluate(x_val, y_val) + self.right.evaluate(x_val, y_val)
        elif self.op == Op.SUB: return self.left.evaluate(x_val, y_val) - self.right.evaluate(x_val, y_val)
        elif self.op == Op.MUL: return self.left.evaluate(x_val, y_val) * self.right.evaluate(x_val, y_val)
        elif self.op == Op.DIV:
            r = self.right.evaluate(x_val, y_val)
            return self.left.evaluate(x_val, y_val) / r if r != 0 else float('inf')
        elif self.op == Op.POW:
            return self.left.evaluate(x_val, y_val) ** self.right.evaluate(x_val, y_val)
        elif self.op == Op.NEG: return -self.left.evaluate(x_val, y_val)
        elif self.op == Op.SIN: return math.sin(self.left.evaluate(x_val, y_val))
        elif self.op == Op.COS: return math.cos(self.left.evaluate(x_val, y_val))
        elif self.op == Op.TAN: return math.tan(self.left.evaluate(x_val, y_val))
        elif self.op == Op.LN: return math.log(max(1e-10, self.left.evaluate(x_val, y_val)))
        elif self.op == Op.EXP: return math.exp(min(100, self.left.evaluate(x_val, y_val)))
        elif self.op == Op.SQRT: return math.sqrt(max(0, self.left.evaluate(x_val, y_val)))
        elif self.op == Op.ABS: return abs(self.left.evaluate(x_val, y_val))
        return 0


# ============================================================================
# CONSTRUCTORS
# ============================================================================

def Const(v): return Expr(Op.CONST, value=float(v))
def VarX(): return Expr(Op.VAR_X)
def VarY(): return Expr(Op.VAR_Y)
def VarZ(): return Expr(Op.VAR_Z)
def Add(l, r): return Expr(Op.ADD, left=l, right=r)
def Sub(l, r): return Expr(Op.SUB, left=l, right=r)
def Mul(l, r): return Expr(Op.MUL, left=l, right=r)
def Div(l, r): return Expr(Op.DIV, left=l, right=r)
def Pow(b, e): return Expr(Op.POW, left=b, right=e)
def Neg(e): return Expr(Op.NEG, left=e)
def Eq(l, r): return Expr(Op.EQ, left=l, right=r)
def Gt(l, r): return Expr(Op.GT, left=l, right=r)
def Lt(l, r): return Expr(Op.LT, left=l, right=r)
def Gte(l, r): return Expr(Op.GTE, left=l, right=r)
def Lte(l, r): return Expr(Op.LTE, left=l, right=r)
def System(e1, e2): return Expr(Op.SYSTEM, left=e1, right=e2)
def Or(l, r): return Expr(Op.OR, left=l, right=r)
def And(l, r): return Expr(Op.AND, left=l, right=r)
def Deriv(e): return Expr(Op.DERIV, left=e)
def Sin(e): return Expr(Op.SIN, left=e)
def Cos(e): return Expr(Op.COS, left=e)
def Tan(e): return Expr(Op.TAN, left=e)
def Ln(e): return Expr(Op.LN, left=e)
def Exp(e): return Expr(Op.EXP, left=e)
def Sqrt(e): return Expr(Op.SQRT, left=e)
def Abs(e): return Expr(Op.ABS, left=e)


# ============================================================================
# PARSER
# ============================================================================

class MathParser:
    """Parse string/LaTeX to expression tree."""
    
    def __init__(self):
        self.tokens = []
        self.pos = 0
    
    def parse(self, text: str) -> Expr:
        text = self._preprocess(text)
        
        if self._is_derivative_request(text):
            return self._parse_derivative_request(text)
        
        if ',' in text or ';' in text:
            return self._parse_system(text)
        
        for rel_op in ['>=', '<=', '!=', '=', '>', '<']:
            if rel_op in text:
                return self._parse_relation(text, rel_op)
        
        return self._parse_expression(text)
    
    def _preprocess(self, text: str) -> str:
        text = text.replace('\\cdot', '*').replace('\\times', '*').replace('\\div', '/')
        frac_pattern = r'\\frac\{([^}]*)\}\{([^}]*)\}'
        while re.search(frac_pattern, text):
            text = re.sub(frac_pattern, r'(\1)/(\2)', text)
        text = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', text)
        text = re.sub(r'e\^{([^}]*)}', r'exp(\1)', text)
        text = re.sub(r'\^{([^}]*)}', r'^(\1)', text)
        for func in ['sin', 'cos', 'tan', 'ln', 'log', 'exp']:
            text = text.replace(f'\\{func}', func)
        return text.strip()
    
    def _is_derivative_request(self, text: str) -> bool:
        patterns = [r'd/dx', r'derivative\s+of', r'differentiate']
        return any(re.search(p, text.lower()) for p in patterns)
    
    def _parse_derivative_request(self, text: str) -> Expr:
        match = re.search(r'd/d[xyz]\s*\((.+)\)', text)
        if match:
            expr_str = match.group(1)
        else:
            match = re.search(r'(?:derivative\s+of|differentiate)\s+(.+)', text, re.I)
            expr_str = match.group(1) if match else text
        return Deriv(self._parse_expression(expr_str))
    
    def _parse_system(self, text: str) -> Expr:
        parts = re.split(r'[,;]', text)
        exprs = [self.parse(p.strip()) for p in parts if p.strip()]
        result = exprs[0]
        for expr in exprs[1:]:
            result = System(result, expr)
        return result
    
    def _parse_relation(self, text: str, rel_op: str) -> Expr:
        parts = text.split(rel_op, 1)
        left = self._parse_expression(parts[0])
        right = self._parse_expression(parts[1])
        op_map = {'=': Eq, '>': Gt, '<': Lt, '>=': Gte, '<=': Lte}
        return op_map[rel_op](left, right)
    
    def _parse_expression(self, text: str) -> Expr:
        self._tokenize(text)
        self.pos = 0
        return self._parse_additive()
    
    def _tokenize(self, text: str):
        self.tokens = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            if text[i:i+2] in ['>=', '<=', '!=', '**']:
                self.tokens.append(text[i:i+2])
                i += 2
                continue
            for func in ['sin', 'cos', 'tan', 'ln', 'log', 'exp', 'sqrt', 'abs']:
                if text[i:i+len(func)].lower() == func:
                    self.tokens.append(func)
                    i += len(func)
                    break
            else:
                if text[i].isdigit() or (text[i] == '.' and i+1 < len(text) and text[i+1].isdigit()):
                    j = i
                    while j < len(text) and (text[j].isdigit() or text[j] == '.'):
                        j += 1
                    self.tokens.append(float(text[i:j]))
                    i = j
                elif text[i].isalpha():
                    self.tokens.append(text[i])
                    i += 1
                else:
                    self.tokens.append(text[i])
                    i += 1
    
    def _current(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    
    def _consume(self, expected=None):
        token = self._current()
        self.pos += 1
        return token
    
    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while self._current() in ['+', '-']:
            op = self._consume()
            right = self._parse_multiplicative()
            left = Add(left, right) if op == '+' else Sub(left, right)
        return left
    
    def _parse_multiplicative(self) -> Expr:
        left = self._parse_power()
        while self._current() in ['*', '/', '·']:
            op = self._consume()
            right = self._parse_power()
            left = Mul(left, right) if op != '/' else Div(left, right)
        return left
    
    def _parse_power(self) -> Expr:
        left = self._parse_unary()
        if self._current() in ['^', '**']:
            self._consume()
            right = self._parse_power()
            if left.op == Op.CONST and abs(left.value - math.e) < 0.01:
                return Exp(right)
            return Pow(left, right)
        return left
    
    def _parse_unary(self) -> Expr:
        if self._current() == '-':
            self._consume()
            return Neg(self._parse_unary())
        if self._current() == '+':
            self._consume()
        return self._parse_primary()
    
    def _parse_primary(self) -> Expr:
        token = self._current()
        
        if isinstance(token, (int, float)):
            self._consume()
            if self._current() in ['x', 'y', 'z', '(']:
                right = self._parse_primary()
                return Mul(Const(token), right)
            return Const(token)
        
        if token in ['(', '{', '[']:
            self._consume()
            expr = self._parse_additive()
            if self._current() in [')', '}', ']']:
                self._consume()
            return expr
        
        if token in ['sin', 'cos', 'tan', 'ln', 'log', 'exp', 'sqrt', 'abs']:
            self._consume()
            if self._current() in ['(', '{']:
                self._consume()
                arg = self._parse_additive()
                if self._current() in [')', '}']:
                    self._consume()
            else:
                arg = self._parse_primary()
            func_map = {'sin': Sin, 'cos': Cos, 'tan': Tan, 'ln': Ln, 'log': Ln, 'exp': Exp, 'sqrt': Sqrt, 'abs': Abs}
            return func_map[token](arg)
        
        if token == 'x':
            self._consume()
            return VarX()
        if token == 'y':
            self._consume()
            return VarY()
        if token == 'z':
            self._consume()
            return VarZ()
        if token == 'e':
            self._consume()
            if self._current() == '^':
                self._consume()
                return Exp(self._parse_primary())
            return Const(math.e)
        if token == 'pi' or token == 'π':
            self._consume()
            return Const(math.pi)
        
        if token is not None:
            self._consume()
        return Const(0)


# ============================================================================
# SIMPLIFIER
# ============================================================================

class Simplifier:
    """Simplify expressions."""
    
    def simplify(self, expr: Expr) -> Expr:
        if expr is None:
            return None
        if expr.left:
            expr.left = self.simplify(expr.left)
        if expr.right:
            expr.right = self.simplify(expr.right)
        expr = self._constant_fold(expr)
        expr = self._identity_rules(expr)
        expr = self._zero_rules(expr)
        return expr
    
    def _constant_fold(self, expr: Expr) -> Expr:
        if expr.op == Op.ADD and expr.left.op == Op.CONST and expr.right.op == Op.CONST:
            return Const(expr.left.value + expr.right.value)
        if expr.op == Op.SUB and expr.left.op == Op.CONST and expr.right.op == Op.CONST:
            return Const(expr.left.value - expr.right.value)
        if expr.op == Op.MUL and expr.left.op == Op.CONST and expr.right.op == Op.CONST:
            return Const(expr.left.value * expr.right.value)
        if expr.op == Op.DIV and expr.left.op == Op.CONST and expr.right.op == Op.CONST and expr.right.value != 0:
            return Const(expr.left.value / expr.right.value)
        if expr.op == Op.POW and expr.left.op == Op.CONST and expr.right.op == Op.CONST:
            return Const(expr.left.value ** expr.right.value)
        if expr.op == Op.NEG and expr.left.op == Op.CONST:
            return Const(-expr.left.value)
        return expr
    
    def _identity_rules(self, expr: Expr) -> Expr:
        if expr.op == Op.ADD:
            if expr.right.op == Op.CONST and expr.right.value == 0:
                return expr.left
            if expr.left.op == Op.CONST and expr.left.value == 0:
                return expr.right
        if expr.op == Op.MUL:
            if expr.right.op == Op.CONST and expr.right.value == 1:
                return expr.left
            if expr.left.op == Op.CONST and expr.left.value == 1:
                return expr.right
        if expr.op == Op.POW:
            if expr.right.op == Op.CONST and expr.right.value == 1:
                return expr.left
            if expr.right.op == Op.CONST and expr.right.value == 0:
                return Const(1)
        if expr.op == Op.NEG and expr.left.op == Op.NEG:
            return expr.left.left
        return expr
    
    def _zero_rules(self, expr: Expr) -> Expr:
        if expr.op == Op.MUL:
            if (expr.left.op == Op.CONST and expr.left.value == 0) or \
               (expr.right.op == Op.CONST and expr.right.value == 0):
                return Const(0)
        if expr.op == Op.DIV and expr.left.op == Op.CONST and expr.left.value == 0:
            return Const(0)
        return expr


# ============================================================================
# DIFFERENTIATOR
# ============================================================================

class Differentiator:
    """Symbolic differentiation."""
    
    def __init__(self):
        self.simplifier = Simplifier()
    
    def differentiate(self, expr: Expr, var: Op = Op.VAR_X) -> Expr:
        result = self._diff(expr, var)
        return self.simplifier.simplify(result)
    
    def _diff(self, expr: Expr, var: Op) -> Expr:
        if expr.op == Op.CONST:
            return Const(0)
        if expr.op == var:
            return Const(1)
        if expr.op in [Op.VAR_X, Op.VAR_Y, Op.VAR_Z] and expr.op != var:
            return Const(0)
        
        if expr.op == Op.ADD:
            return Add(self._diff(expr.left, var), self._diff(expr.right, var))
        if expr.op == Op.SUB:
            return Sub(self._diff(expr.left, var), self._diff(expr.right, var))
        if expr.op == Op.MUL:
            f, g = expr.left, expr.right
            return Add(Mul(self._diff(f, var), g.copy()), Mul(f.copy(), self._diff(g, var)))
        if expr.op == Op.DIV:
            f, g = expr.left, expr.right
            num = Sub(Mul(self._diff(f, var), g.copy()), Mul(f.copy(), self._diff(g, var)))
            return Div(num, Pow(g.copy(), Const(2)))
        if expr.op == Op.POW and expr.right.op == Op.CONST:
            n = expr.right.value
            return Mul(Mul(Const(n), Pow(expr.left.copy(), Const(n-1))), self._diff(expr.left, var))
        if expr.op == Op.NEG:
            return Neg(self._diff(expr.left, var))
        if expr.op == Op.SIN:
            return Mul(Cos(expr.left.copy()), self._diff(expr.left, var))
        if expr.op == Op.COS:
            return Mul(Neg(Sin(expr.left.copy())), self._diff(expr.left, var))
        if expr.op == Op.LN:
            return Div(self._diff(expr.left, var), expr.left.copy())
        if expr.op == Op.EXP:
            return Mul(Exp(expr.left.copy()), self._diff(expr.left, var))
        if expr.op == Op.SQRT:
            return Div(self._diff(expr.left, var), Mul(Const(2), Sqrt(expr.left.copy())))
        return Const(0)


# ============================================================================
# SOLVERS
# ============================================================================

class LinearSolver:
    """Solve ax + b = c."""
    
    def solve(self, equation: Expr) -> Optional[Dict]:
        if equation.op != Op.EQ:
            return None
        lhs = Sub(equation.left, equation.right)
        a, b = self._collect(lhs)
        if abs(a) < 1e-10:
            return None
        return {'x': -b / a}
    
    def _collect(self, expr: Expr) -> Tuple[float, float]:
        if expr.op == Op.CONST:
            return (0, expr.value)
        if expr.op == Op.VAR_X:
            return (1, 0)
        if expr.op == Op.NEG:
            a, b = self._collect(expr.left)
            return (-a, -b)
        if expr.op == Op.ADD:
            a1, b1 = self._collect(expr.left)
            a2, b2 = self._collect(expr.right)
            return (a1 + a2, b1 + b2)
        if expr.op == Op.SUB:
            a1, b1 = self._collect(expr.left)
            a2, b2 = self._collect(expr.right)
            return (a1 - a2, b1 - b2)
        if expr.op == Op.MUL:
            if expr.left.op == Op.CONST:
                a, b = self._collect(expr.right)
                return (expr.left.value * a, expr.left.value * b)
            if expr.right.op == Op.CONST:
                a, b = self._collect(expr.left)
                return (expr.right.value * a, expr.right.value * b)
        return (0, 0)


class QuadraticSolver:
    """Solve ax² + bx + c = 0."""
    
    def solve(self, equation: Expr) -> Optional[Dict]:
        if equation.op != Op.EQ:
            return None
        lhs = Sub(equation.left, equation.right)
        coeffs = self._collect_terms(lhs)
        a, b, c = coeffs.get(2, 0), coeffs.get(1, 0), coeffs.get(0, 0)
        
        if abs(a) < 1e-10:
            if abs(b) > 1e-10:
                return {'x': [-c / b]}
            return None
        
        disc = b*b - 4*a*c
        if disc < 0:
            real = -b / (2*a)
            imag = math.sqrt(-disc) / (2*a)
            return {'x': [complex(real, imag), complex(real, -imag)]}
        
        sqrt_disc = math.sqrt(disc)
        x1 = (-b + sqrt_disc) / (2*a)
        x2 = (-b - sqrt_disc) / (2*a)
        
        if abs(x1 - x2) < 1e-10:
            return {'x': [x1]}
        return {'x': sorted([x1, x2])}
    
    def _collect_terms(self, node: Expr, mult: float = 1.0) -> Dict[int, float]:
        coeffs = {}
        def add(p, c):
            coeffs[p] = coeffs.get(p, 0) + c
        
        def collect(node, m):
            if node is None:
                return
            if node.op == Op.CONST:
                add(0, m * node.value)
            elif node.op == Op.VAR_X:
                add(1, m)
            elif node.op == Op.ADD:
                collect(node.left, m)
                collect(node.right, m)
            elif node.op == Op.SUB:
                collect(node.left, m)
                collect(node.right, -m)
            elif node.op == Op.NEG:
                collect(node.left, -m)
            elif node.op == Op.MUL:
                if node.left.op == Op.CONST:
                    collect(node.right, m * node.left.value)
                elif node.right.op == Op.CONST:
                    collect(node.left, m * node.right.value)
                elif node.left.op == Op.VAR_X and node.right.op == Op.VAR_X:
                    add(2, m)
            elif node.op == Op.POW:
                if node.left.op == Op.VAR_X and node.right.op == Op.CONST:
                    add(int(node.right.value), m)
        
        collect(node, mult)
        return coeffs


class SystemSolver:
    """Solve 2x2 linear systems."""
    
    def solve(self, system: Expr) -> Optional[Dict]:
        equations = self._flatten(system)
        if len(equations) != 2:
            return None
        
        coeffs = []
        for eq in equations:
            if eq.op != Op.EQ:
                return None
            lhs = Sub(eq.left, eq.right)
            a, b, c = self._extract(lhs)
            coeffs.append((a, b, -c))
        
        a1, b1, c1 = coeffs[0]
        a2, b2, c2 = coeffs[1]
        
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            return None
        
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return {'x': x, 'y': y}
    
    def _flatten(self, expr: Expr) -> List[Expr]:
        if expr.op == Op.SYSTEM:
            return self._flatten(expr.left) + self._flatten(expr.right)
        return [expr]
    
    def _extract(self, expr: Expr) -> Tuple[float, float, float]:
        a, b, c = 0, 0, 0
        def collect(node, m=1):
            nonlocal a, b, c
            if node.op == Op.CONST:
                c += m * node.value
            elif node.op == Op.VAR_X:
                a += m
            elif node.op == Op.VAR_Y:
                b += m
            elif node.op == Op.ADD:
                collect(node.left, m)
                collect(node.right, m)
            elif node.op == Op.SUB:
                collect(node.left, m)
                collect(node.right, -m)
            elif node.op == Op.NEG:
                collect(node.left, -m)
            elif node.op == Op.MUL:
                if node.left.op == Op.CONST:
                    collect(node.right, m * node.left.value)
                elif node.right.op == Op.CONST:
                    collect(node.left, m * node.right.value)
        collect(expr)
        return (a, b, c)


class InequalitySolver:
    """Solve linear/quadratic inequalities."""
    
    def solve(self, ineq: Expr) -> Optional[Dict]:
        if ineq.op not in [Op.GT, Op.LT, Op.GTE, Op.LTE]:
            return None
        lhs = Sub(ineq.left, ineq.right)
        result = self._solve_linear(lhs, ineq.op)
        if result:
            return result
        return self._solve_quadratic(lhs, ineq.op)
    
    def _solve_linear(self, expr: Expr, rel_op: Op) -> Optional[Dict]:
        a, b = 0, 0
        def collect(node, m=1):
            nonlocal a, b
            if node.op == Op.CONST:
                b += m * node.value
            elif node.op == Op.VAR_X:
                a += m
            elif node.op == Op.ADD:
                collect(node.left, m)
                collect(node.right, m)
            elif node.op == Op.SUB:
                collect(node.left, m)
                collect(node.right, -m)
            elif node.op == Op.NEG:
                collect(node.left, -m)
            elif node.op == Op.MUL:
                if node.left.op == Op.CONST:
                    collect(node.right, m * node.left.value)
                elif node.right.op == Op.CONST:
                    collect(node.left, m * node.right.value)
        collect(expr)
        
        if abs(a) < 1e-10:
            return None
        
        boundary = -b / a
        if a > 0:
            if rel_op in [Op.GT, Op.GTE]:
                return {'type': 'linear', 'solution': f'x {">" if rel_op == Op.GT else ">="} {boundary:.4g}'}
            else:
                return {'type': 'linear', 'solution': f'x {"<" if rel_op == Op.LT else "<="} {boundary:.4g}'}
        else:
            if rel_op in [Op.GT, Op.GTE]:
                return {'type': 'linear', 'solution': f'x {"<" if rel_op == Op.GT else "<="} {boundary:.4g}'}
            else:
                return {'type': 'linear', 'solution': f'x {">" if rel_op == Op.LT else ">="} {boundary:.4g}'}
    
    def _solve_quadratic(self, expr: Expr, rel_op: Op) -> Optional[Dict]:
        eq = Eq(expr, Const(0))
        solver = QuadraticSolver()
        result = solver.solve(eq)
        if not result or 'x' not in result:
            return None
        
        roots = result['x']
        if isinstance(roots[0], complex):
            val = expr.evaluate(0)
            if rel_op in [Op.GT, Op.GTE]:
                return {'type': 'quadratic', 'solution': 'all x' if val > 0 else 'no solution'}
            else:
                return {'type': 'quadratic', 'solution': 'all x' if val < 0 else 'no solution'}
        
        roots = sorted([float(r.real) if isinstance(r, complex) else r for r in roots])
        if len(roots) == 1:
            return {'type': 'quadratic', 'solution': f'x ≠ {roots[0]:.4g}' if rel_op in [Op.GT, Op.LT] else f'x = {roots[0]:.4g}'}
        
        r1, r2 = roots
        leading_pos = expr.evaluate(1000) > 0
        
        if rel_op in [Op.GT, Op.GTE]:
            if leading_pos:
                return {'type': 'quadratic', 'solution': f'x < {r1:.4g} or x > {r2:.4g}'}
            else:
                return {'type': 'quadratic', 'solution': f'{r1:.4g} < x < {r2:.4g}'}
        else:
            if leading_pos:
                return {'type': 'quadratic', 'solution': f'{r1:.4g} < x < {r2:.4g}'}
            else:
                return {'type': 'quadratic', 'solution': f'x < {r1:.4g} or x > {r2:.4g}'}


class MathSolver:
    """Unified solver."""
    
    def __init__(self):
        self.parser = MathParser()
        self.simplifier = Simplifier()
        self.differentiator = Differentiator()
        self.linear = LinearSolver()
        self.quadratic = QuadraticSolver()
        self.system = SystemSolver()
        self.inequality = InequalitySolver()
    
    def solve(self, problem: Union[str, Expr]) -> Dict:
        result = {'input': str(problem)}
        try:
            expr = self.parser.parse(problem) if isinstance(problem, str) else problem
            result['parsed'] = str(expr)
            ptype = self._classify(expr)
            result['type'] = ptype.name
            
            if ptype == ProblemType.DERIVATIVE:
                inner = expr.left if expr.op == Op.DERIV else expr
                deriv = self.differentiator.differentiate(inner)
                result['solution'] = {'derivative': str(deriv)}
            elif ptype == ProblemType.LINEAR_EQ:
                result['solution'] = self.linear.solve(expr)
            elif ptype == ProblemType.QUADRATIC_EQ:
                result['solution'] = self.quadratic.solve(expr)
            elif ptype == ProblemType.SYSTEM_EQ:
                result['solution'] = self.system.solve(expr)
            elif ptype in [ProblemType.LINEAR_INEQ, ProblemType.QUADRATIC_INEQ]:
                result['solution'] = self.inequality.solve(expr)
            else:
                result['solution'] = None
        except Exception as e:
            result['error'] = str(e)
        return result
    
    def _classify(self, expr: Expr) -> ProblemType:
        if expr.op == Op.DERIV:
            return ProblemType.DERIVATIVE
        if expr.op == Op.SYSTEM:
            return ProblemType.SYSTEM_EQ
        if expr.op in [Op.GT, Op.LT, Op.GTE, Op.LTE]:
            return ProblemType.QUADRATIC_INEQ if self._has_pow2(expr) else ProblemType.LINEAR_INEQ
        if expr.op == Op.EQ:
            if self._has_pow2(expr):
                return ProblemType.QUADRATIC_EQ
            if self._has_trig(expr):
                return ProblemType.TRIG_EQ
            return ProblemType.LINEAR_EQ
        return ProblemType.UNKNOWN
    
    def _has_pow2(self, expr: Expr) -> bool:
        if expr is None:
            return False
        if expr.op == Op.POW and expr.left and expr.left.op == Op.VAR_X:
            if expr.right and expr.right.op == Op.CONST and expr.right.value == 2:
                return True
        return self._has_pow2(expr.left) or self._has_pow2(expr.right)
    
    def _has_trig(self, expr: Expr) -> bool:
        if expr is None:
            return False
        if expr.op in [Op.SIN, Op.COS, Op.TAN]:
            return True
        return self._has_trig(expr.left) or self._has_trig(expr.right)


# ============================================================================
# PROBLEM GENERATOR
# ============================================================================

class ProblemGenerator:
    """Generate problems for testing/training."""
    
    @staticmethod
    def linear_equation(solution: int = None):
        if solution is None:
            solution = random.randint(-10, 10)
        a = random.choice([1, 2, 3, -1, -2, -3])
        b = random.randint(-10, 10)
        c = a * solution + b
        lhs = Add(Mul(Const(a), VarX()), Const(b))
        return Eq(lhs, Const(c)), {'type': 'linear_eq', 'solution': solution}
    
    @staticmethod
    def quadratic_equation(r1: int = None, r2: int = None):
        if r1 is None:
            r1 = random.randint(-5, 5)
        if r2 is None:
            r2 = random.randint(-5, 5)
        b, c = -(r1 + r2), r1 * r2
        x2 = Pow(VarX(), Const(2))
        lhs = x2
        if b != 0:
            lhs = Add(lhs, Mul(Const(b), VarX()))
        if c != 0:
            lhs = Add(lhs, Const(c))
        return Eq(lhs, Const(0)), {'type': 'quadratic_eq', 'roots': tuple(sorted([r1, r2]))}
    
    @staticmethod
    def system_equation(x_sol: int = None, y_sol: int = None):
        if x_sol is None:
            x_sol = random.randint(-5, 5)
        if y_sol is None:
            y_sol = random.randint(-5, 5)
        a1, b1 = random.randint(1, 3), random.randint(-2, 2)
        a2, b2 = random.randint(-2, 2), random.randint(1, 3)
        c1 = a1 * x_sol + b1 * y_sol
        c2 = a2 * x_sol + b2 * y_sol
        eq1 = Eq(Add(Mul(Const(a1), VarX()), Mul(Const(b1), VarY())), Const(c1))
        eq2 = Eq(Add(Mul(Const(a2), VarX()), Mul(Const(b2), VarY())), Const(c2))
        return System(eq1, eq2), {'type': 'system_eq', 'solution': (x_sol, y_sol)}
    
    @staticmethod
    def linear_inequality(boundary: int = None, direction: str = None):
        if boundary is None:
            boundary = random.randint(-5, 5)
        if direction is None:
            direction = random.choice(['gt', 'lt'])
        a = random.choice([1, 2, 3, -1, -2, -3])
        b = random.randint(-10, 10)
        c = a * boundary + b
        lhs = Add(Mul(Const(a), VarX()), Const(b))
        actual_dir = direction
        if a < 0:
            actual_dir = 'lt' if direction == 'gt' else 'gt'
        op = Gt if actual_dir == 'gt' else Lt
        return op(lhs, Const(c)), {'type': 'linear_ineq', 'boundary': boundary, 'direction': direction}
    
    @staticmethod
    def quadratic_inequality(r1: int = None, r2: int = None, direction: str = None):
        if r1 is None:
            r1 = random.randint(-3, 3)
        if r2 is None:
            r2 = random.randint(-3, 3)
        if direction is None:
            direction = random.choice(['gt', 'lt'])
        b, c = -(r1 + r2), r1 * r2
        x2 = Pow(VarX(), Const(2))
        lhs = x2
        if b != 0:
            lhs = Add(lhs, Mul(Const(b), VarX()))
        if c != 0:
            lhs = Add(lhs, Const(c))
        op = Gt if direction == 'gt' else Lt
        return op(lhs, Const(0)), {'type': 'quadratic_ineq', 'roots': tuple(sorted([r1, r2])), 'direction': direction}
    
    @staticmethod
    def invalid_equation():
        kind = random.choice(['div_zero', 'contradiction', 'always_false'])
        if kind == 'div_zero':
            return Eq(Div(VarX(), Const(0)), Const(5)), {'type': 'invalid', 'reason': kind}
        elif kind == 'contradiction':
            return Eq(Add(Mul(Const(0), VarX()), Const(5)), Const(3)), {'type': 'invalid', 'reason': kind}
        else:
            return Lt(Add(Pow(VarX(), Const(2)), Const(1)), Const(0)), {'type': 'invalid', 'reason': kind}
