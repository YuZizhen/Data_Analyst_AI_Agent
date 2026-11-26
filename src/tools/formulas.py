"""Evaluate arithmetic over numeric CSV fields without executing Python code."""

import ast
import operator
from math import isfinite

import pandas as pd
from langchain_core.tools import ToolException

from .dataset import column, finite_numbers

OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def formula_values(frame: pd.DataFrame, expression: str) -> pd.Series:
    expression = expression.translate(str.maketrans({"×": "*", "÷": "/", "−": "-"})).strip()
    if len(expression) > 500:
        raise ToolException("The formula is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ToolException("Use a formula with numeric columns, numbers, and arithmetic operators.") from error
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ToolException("The formula has too many operations.")

    def calculate(node):
        if isinstance(node, ast.Name):
            return finite_numbers(column(frame, node.id)).astype(float)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float} and isfinite(node.value):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = calculate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and (operation := OPERATIONS.get(type(node.op))):
            return operation(calculate(node.left), calculate(node.right))
        raise ToolException("Formulas support only column names, finite numbers, +, -, *, / and parentheses.")

    try:
        value = calculate(tree.body)
    except (ZeroDivisionError, OverflowError) as error:
        raise ToolException("The formula divides by zero or exceeds the numeric range.") from error
    if not isinstance(value, pd.Series):
        raise ToolException("The formula must use at least one column.")
    return finite_numbers(value)
