from Parser import parse, NULL_VALUE, NULL_LITERAL
from Value import Value, ValueType, IntValue, ListValue, StringValue, FloatValue, NullValue
from FunctionTemplate import FunctionTemplate


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

    def get_value(self, key):
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

