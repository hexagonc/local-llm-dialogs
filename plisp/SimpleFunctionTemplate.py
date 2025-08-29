from .FunctionTemplate import FunctionTemplate

class SimpleFunctionTemplate(FunctionTemplate):
    # function_lambda this is a lambda function that takes the function template itself
    # and a list of evaluated arguments as arguments. The function should return a Value object.
    # The lambda function will not have access to the evaluation environment.
    def __init__(self, name, function_lambda):
        super().__init__(name, function_lambda)

    def copy(self):
        return SimpleFunctionTemplate(self.name, self.function_lambda)

    def evaluate(self, evaluation_env = None):
        evaluated_args = [arg.evaluate(evaluation_env) for arg in self.actual_arguments]
        return self.function_lambda(self, evaluated_args)