from .Parser import parse
from .Value import NULL_VALUE, NULL_LITERAL
from .FunctionTemplate import FunctionTemplate


class Environment:
    def __init__(self, parent = None):
        self.parent = parent
        self.var_map = {}
        self.function_map = {}
        self.documentation_map = {}

    def evaluate(self, value):
        return value.evaluate(self)

    def map_value(self, key, value):
        self.var_map[key] = value
        return value

    def unbind_value(self, key):
        if key in self.var_map:
            self.var_map.pop(key)

    def get_value(self, key):
        if key == NULL_LITERAL:
            return NULL_VALUE
        if key in self.var_map:
            return self.var_map[key]
        if self.parent is not None:
            return self.parent.get_value(key)
        raise Exception(f"variable not defined: {key}")

    def has_value(self, key):
        if key == NULL_LITERAL:
            return NULL_VALUE
        if key in self.var_map:
            return self.var_map[key]
        if self.parent is not None:
            return self.parent.get_value(key)
        return None

    # function_lambda is a lambda function that takes the function template itself and the
    # evaluation environment as arguments. The function should return a Value object.
    def map_function(self, key, function_lambda, documentation = None):
        self.function_map[key] = FunctionTemplate(key, function_lambda)
        if documentation is not None:
            self.documentation_map[key] = documentation
        return self

    def map_function_template(self, function_template):
        self.function_map[function_template.name] = function_template
        return function_template

    def get_function(self, key):
        if key in self.function_map:
            return self.function_map[key].copy()
        if self.parent is not None:
            return self.parent.get_function(key)
        return None

    def evaluate_exp(self, exp):
        out = None
        for results in parse(exp, True):
            out = self.evaluate(results)
        return out

    def serialize(self):
        item_separator = "~*.*~"
        pair_separator = "+~.~+"
        parent_separator = "+..+"
        if self.parent:
            serialized_parent = self.parent.serialize() + parent_separator
        else:
            serialized_parent = ""
        inner = item_separator.join([f"{kv_pair[0]}{pair_separator}{kv_pair[1].serialize()}" for kv_pair in self.var_map.items()])
        return serialized_parent + inner

    def fromSerialized(self, serialized:str):
        from .LispTools import add_basic_functions, add_arithmetic_functions
        from .FileTools import add_filesystem_functions
        item_separator = "~*.*~"
        pair_separator = "+~.~+"
        parent_separator = "+..+"

        last = serialized.rfind(parent_separator)
        if last >= 0:
            parent_env = Environment()

            self.parent = parent_env
            parent_env.fromSerialized(serialized[0:last])
            serialized = serialized[len(parent_separator):]
        else:
            add_basic_functions(self)
            add_arithmetic_functions(self)
            add_filesystem_functions(self)
        for serialized_kv in serialized.split(item_separator):
            key, serialized_value = serialized_kv.split(pair_separator)
            self.map_value(key, self.evaluate_exp(serialized_value))

