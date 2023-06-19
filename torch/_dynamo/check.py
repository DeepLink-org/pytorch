import ast
import astunparse
import torch
import functools

from .symbolic_convert import InstructionTranslatorBase, Instruction
from .exc import unimplemented, TorchDynamoException

from . import (
    variables,
)

class visitor(ast.NodeVisitor):
    def __init__(self, name_map) -> None:
        super().__init__()
        self.name_map = name_map
    def visit_Name(self, node):
        if node.id in self.name_map:
            node.id = self.name_map[node.id]


pass_thru = None
pass_cond = None

extra_guards = None

global_scope = None

reverse_locals = None

blacklist = dict()
# import os
# cwd = os.getcwd()
with open("blacklist", "r") as f:
    read_lines = f.readlines()
    for line in read_lines:
        components = line.strip().split(':')
        func = eval(components[0][:components[0].index('(')])
        if len(components) == 1:
            blacklist[func] = ''
        else:
            blacklist[func] = components

def combine_scopes(left, right):
    if left is None:
        return right

    if right is None:
        return left

    return {**left, **right}

class Partialsupported(TorchDynamoException):
    def __init__(self):
        super().__init__()

def partial_implemented():
    raise Partialsupported()

def check(fn:variables.VariableTracker, args, kwargs, scope):
    global pass_thru
    global pass_cond
    global extra_guards

    name = get_fn_name(fn, scope)
    if not name.startswith("torch."):
        return

    if name is None:
        unimplemented("Backend has not supported this op")

    func = eval(name, global_scope, scope)

    if func in blacklist:
        if blacklist[func] == '':
            unimplemented("Backend has not supported this op")
        else:
            name_map = dict()
            func_str = blacklist[func][0]
            args_cond = blacklist[func][1:]
            body = ast.parse(func_str).body[0]
            local_env = dict()
            for i in range(len(args)):
                local_env[body.value.args[i].id] = get_arg_value(args[i], scope)
                if args[i] in reverse_locals:
                    name_map[body.value.args[i].id] = reverse_locals[args[i]]
            for key in kwargs:
                local_env[key] = get_arg_value(kwargs[key], scope)
                if kwargs[key] in reverse_locals:
                    name_map[key] = reverse_locals[kwargs[key]]

            if isinstance(fn, variables.GetAttrVariable) or isinstance(fn, variables.UserMethodVariable):
                if not (isinstance(fn, variables.PythonModuleVariable) or isinstance(fn, variables.TorchVariable) or isinstance(fn, variables.UserDefinedClassVariable)):
                    v_self = get_arg_value(fn.obj, scope)
                    local_env['self'] = v_self

            if pass_thru == func:
                assert pass_cond is not None
                cond_ast = ast.parse(pass_cond)
                v = visitor(name_map)
                v.visit(cond_ast)
                new_cond = astunparse.unparse(cond_ast).strip()
                extra_guards = new_cond
                pass_thru = None
                pass_cond = None
                return

            if scope != global_scope:
                local_env = combine_scopes(scope, local_env)
            result = False
            for cond in args_cond:
                result = eval(cond, global_scope, local_env)
                if result:
                    pass_thru = func
                    pass_cond = cond
                    partial_implemented()

            unimplemented("Backend has not supported this op")


from . import source
def get_fn_name(fn:variables.VariableTracker, scope):
    if isinstance(fn, variables.BuiltinVariable):
        return fn.fn.__qualname__
    elif isinstance(fn, variables.NNModuleVariable):
        m = fn.python_type().__module__
        if m == '__main__':
            return fn.python_type().__qualname__
        return m + '.' + fn.python_type().__qualname__
    elif isinstance(fn, variables.UserMethodVariable):
        return get_fn_name(fn.obj, scope) + '.' + fn.get_name()
    elif isinstance(fn, variables.functions.BaseUserFunctionVariable):
        return fn.get_name()
    elif isinstance(fn, variables.GetAttrVariable):
        return get_fn_name(fn.obj, scope) + '.' + fn.name
    elif isinstance(fn, variables.TorchVariable):
        return fn.value.__module__ + '.' + fn.value.__name__
    elif isinstance(fn, variables.TensorVariable):
        m = fn.python_type().__module__
        if m == '__main__':
            return fn.python_type().__qualname__
        return m + '.' + fn.python_type().__qualname__

    if fn.source is not None:
        name = fn.source.name()
        if isinstance(fn, variables.GetAttrVariable):
            name = eval(name + '.__qualname__', global_scope, scope)
        return name

    return None

def get_arg_value(var:variables.VariableTracker, scope):

    if var.source is not None:
        name = var.source.name()
        return eval(name, global_scope, scope)

    if isinstance(var, variables.TensorVariable):
        return var.as_proxy().node.meta["example_value"]
    elif isinstance(var, variables.ConstantVariable):
        return var.as_proxy()
    elif isinstance(var, variables.GetAttrVariable):
        return getattr(get_arg_value(var.obj, scope), var.name)
    elif isinstance(var, variables.BaseListVariable):
        return var.python_type()([get_arg_value(x, scope) for x in var.items])
    elif isinstance(var, variables.ConstDictVariable):
        return {k: get_arg_value(v, scope) for k, v in var.items.items()}
    elif isinstance(var, variables.SymNodeVariable):
        return var.sym_num
    else:
        if hasattr(var, "value"):
            return var.value
        if hasattr(var, "fn"):
            return var.fn
        unimplemented()