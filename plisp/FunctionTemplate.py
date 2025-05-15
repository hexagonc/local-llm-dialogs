
# function_lambda this is a lambda function that takes the function template itself
# and the evaluation environment as arguments. The function should return a Value object.

class FunctionTemplate:
    def __init__(self, name, function_lambda):
        self.name = name
        self.actual_arguments = None
        self.num_arguments = 0
        self.function_lambda = function_lambda

    def set_actual_arguments(self, arg_list):
        self.actual_arguments = [a.copy() for a in arg_list]
        self.num_arguments = len(self.actual_arguments)
        return self

    def copy(self):
        return FunctionTemplate(self.name, self.function_lambda)


    def evaluate(self, evaluation_env = None):
        return self.function_lambda(self, evaluation_env)