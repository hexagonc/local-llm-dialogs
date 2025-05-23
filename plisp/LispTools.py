import time

from .Value import Value, ListValue, NULL_VALUE, IntValue, StringValue, FloatValue
from .Environment import Environment
from .SimpleFunctionTemplate import SimpleFunctionTemplate
from .HashtableValue import StringHashtableValue, IntHashtableValue



class LispTools:
    @staticmethod
    def make_integer(intValue):
        return IntValue(intValue)

    @staticmethod
    def make_list(values):
        return ListValue(values)

    @staticmethod
    def make_str(string):
        return StringValue(string)

    @staticmethod
    def make_symbol(string):
        return StringValue(string, is_symbol=True)

def add_basic_functions(env:Environment):

    def get_datetime_string(template, evaluated_args):
        from datetime import datetime

        arg_length = len(evaluated_args)
        use_am_pm = True

        if arg_length > 0:
            epoch_milli = evaluated_args[0]
            epoch_seconds = epoch_milli / 1000
            if arg_length > 1:
                use_am_pm_arg:Value = evaluated_args[1]
                use_am_pm = not use_am_pm_arg.is_null()
        else:
            epoch_seconds = time.time()
        dt = datetime.fromtimestamp(epoch_seconds)
        if use_am_pm:
            # Format the datetime object into the desired string format
            formatted_time = dt.strftime("%a %B %d %Y %I:%M:%S %p")
        else:
            formatted_time = dt.strftime("%a %B %d %Y %H:%M:%S")
        return LispTools.make_str(formatted_time)

    env.map_function_template(SimpleFunctionTemplate("get-datetime-string", get_datetime_string))

    def to_string(template, evaluated_args):
        if evaluated_args[0].is_string():
            return evaluated_args[0]
        else:
            return LispTools.make_str(evaluated_args[0].serialize())
        
    env.map_function_template(SimpleFunctionTemplate("string", to_string))

    def if_function(template, evaluation_env):
        condition_pass = not template.actual_arguments[0].evaluate(evaluation_env).is_null()
        if condition_pass:
            return template.actual_arguments[1].evaluate(evaluation_env)
        if len(template.actual_arguments) == 3:
            return template.actual_arguments[2].evaluate(evaluation_env)
        return NULL_VALUE

    env.map_function("if", if_function)

    def setq(template, evaluation_env):
        key = template.actual_arguments[0].string()
        value = template.actual_arguments[1].evaluate(evaluation_env)
        evaluation_env.map_value(key, value)
        return value

    env.map_function("setq", setq)

    def set(template, evaluation_env):
        key = template.actual_arguments[0].string()

        target_env = evaluation_env
        search_env = evaluation_env
        while search_env is not None:
            if key in search_env.var_map:
                target_env = search_env
                break
            search_env = search_env.parent

        value = template.actual_arguments[1].evaluate(evaluation_env)
        target_env.map_value(key, value)
        return value

    env.map_function("set", set)


    def numeric_equals(template, evaluated_args):
        check_numeric_args(evaluated_args)
        lvalue = evaluated_args[0]
        rvalue = evaluated_args[1]
        if lvalue.is_integer() and rvalue.is_integer():
            return to_null(lvalue.int_value() == rvalue.int_value(), lvalue)
        else:
            return to_null(lvalue.float_value() == rvalue.float_value(), lvalue)

    env.map_function_template(SimpleFunctionTemplate("=", numeric_equals))

    def equals(template, evaluated_args):
        lvalue = evaluated_args[0]
        rvalue = evaluated_args[1]
        if lvalue.serialize() == rvalue.serialize():
            return lvalue
        else:
            return NULL_VALUE

    env.map_function_template(SimpleFunctionTemplate("equals", equals))

    def progn(template, evaluation_env):
        result = NULL_VALUE
        for arg in template.actual_arguments:
            result = arg.evaluate(evaluation_env)
        return result

    env.map_function("progn", progn)

    def let(template, evaluation_env):
        # first argument is a binding list
        # remaining arguments are expressions to evaluate sequentially
        # in child environment
        result = NULL_VALUE
        child = Environment(evaluation_env)

        map_binding_list(template.actual_arguments[0], child)

        for arg in template.actual_arguments[1:]:
            result = arg.evaluate(child)
        return result

    env.map_function("let", let)


    def while_function(template, evaluation_env):
        result = NULL_VALUE
        while not template.actual_arguments[0].evaluate(evaluation_env).is_null():
            result = template.actual_arguments[1].evaluate(evaluation_env)
        return result

    env.map_function("while", while_function)

    def for_function(template, evaluation_env):
        # For loop has 4 arguments
        # First argument can be either a variable, x, or a list (i, x) where x is
        # the binding variable for each iteration of the loop and i is the index from
        # 0 to loop count -1
        # second argument is either a list or an integer.  If it is an integer, N, then the loop
        # will iterate from 0 to the N - 1, binding x to each value.  If it is a list, the loop will iterate
        # over the list, binding x to each value.
        # third argument is an expression to evaluate at the end of the loop, defining the return value of the loop
        # fourth argument is the body of the loop, evaluated for each iteration with a different value of x

        # First argument
        loop_params = template.actual_arguments[0] # This is either a list or a symbol
        loop_spec = template.actual_arguments[1] # This is either a list or an integer
        return_exp = template.actual_arguments[2]
        loop_body = template.actual_arguments[3]

        loop_env = Environment(evaluation_env)
        loop_index_var_name = "_"
        if loop_params.is_list():
            loop_index_var_name = loop_params.list()[0].string()
            loop_var_name = loop_params.list()[1]
        else:
            loop_var_name = loop_params.string()

        if loop_spec.is_list():
            L = loop_spec.list()
            max = loop_spec.size()
        else:
            max = loop_spec.int_value()
        i = 0
        def index_iter():
            nonlocal i
            if i < max:
                i += 1
                return i - 1
            return None

        next_index = index_iter()
        while next_index is not None:
            loop_index_value = LispTools.make_integer(next_index)
            loop_env.map_value(loop_index_var_name, loop_index_value)
            loop_value = L[next_index] if loop_spec.is_list() else loop_index_value
            loop_env.map_value(loop_var_name, loop_value)
            return_value = loop_body.evaluate(loop_env)
            next_index = index_iter()
        return return_exp.evaluate(loop_env)

    env.map_function("for", for_function)

    def mapcar(template, evaluation_env):
        # mapcar takes a function and a list of arguments
        # First argument can be either a variable, x, or a list (i, x) where x is
        # the binding variable for each iteration of the loop and i is the index from
        # 0 to loop count -1
        # second argument is either a list or an integer.  If it is an integer, N, then the loop
        # will iterate from 0 to the N - 1, binding x to each value.  If it is a list, the loop will iterate
        # over the list, binding x to each value.
        # third argument is an expression to evaluate at the end of the loop, defining the return value of the loop
        # fourth argument is the body of the loop, evaluated for each iteration with a different value of x
        loop_params = template.actual_arguments[0]  # This is either a list or a symbol
        loop_spec = template.actual_arguments[1]  # This is either a list or an integer
        mapping_body = template.actual_arguments[2] # defines each value in the list

        out = []

        loop_env = Environment(evaluation_env)
        loop_index_var_name = "_"
        if loop_params.is_list():
            loop_index_var_name = loop_params.list()[0].string()
            loop_var_name = loop_params.list()[1]
        else:
            loop_var_name = loop_params.string()

        if loop_spec.is_list():
            L = loop_spec.list()
            max = loop_spec.size()
        else:
            max = loop_spec.int_value()
        i = 0

        def index_iter():
            nonlocal i
            if i < max:
                i += 1
                return i - 1
            return None

        next_index = index_iter()
        while next_index is not None:
            loop_index_value = LispTools.make_integer(next_index)
            loop_env.map_value(loop_index_var_name, loop_index_value)
            loop_value = L[next_index] if loop_spec.is_list() else loop_index_value
            loop_env.map_value(loop_var_name, loop_value)
            return_value = mapping_body.evaluate(loop_env)
            out.append(return_value)
            next_index = index_iter()
        return LispTools.make_list(out)

    env.map_function("mapcar", mapcar)

    def and_funct(template, evaluation_env):
        evaluated_value = NULL_VALUE
        for arg in template.actual_arguments:
            evaluated_value = arg.evaluate(evaluation_env)
            if evaluated_value.is_null():
                return evaluated_value
        return evaluated_value

    env.map_function("and", and_funct)

    def or_funct(template, evaluation_env):
        evaluated_value = NULL_VALUE
        for arg in template.actual_arguments:
            evaluated_value = arg.evaluate(evaluation_env)
            if not evaluated_value.is_null():
                return evaluated_value
        return evaluated_value

    env.map_function("or", or_funct)

    def print_function(template, evaluated_args):
        items = []
        for arg in evaluated_args:
            if arg.is_string():
                items.append(arg.string())

            else:
                items.append(arg.serialize())
        o = " ".join(items)
        print(o)
        return LispTools.make_str(o)

    env.map_function_template(SimpleFunctionTemplate("print", print_function))

    def concat_function(template, evaluated_args):
        items = []
        for arg in evaluated_args:
            items.append(arg.string())
        return LispTools.make_str("".join(items))

    env.map_function_template(SimpleFunctionTemplate("concat", concat_function))

    def list_function(template, evaluated_args):
        return LispTools.make_list([x.copy() for x in evaluated_args])

    env.map_function_template(SimpleFunctionTemplate("list", list_function))

    def make_string_hashtable(template, evaluated_args):
        if len(evaluated_args) == 0:
            return StringHashtableValue(LispTools.make_list([]))
        return StringHashtableValue(evaluated_args[0])

    env.map_function_template(SimpleFunctionTemplate("make-string-hashtable", make_string_hashtable))

    def make_int_hashtable(template, evaluated_args):
        if len(evaluated_args) == 0:
            return IntHashtableValue(LispTools.make_list([]))
        return IntHashtableValue(evaluated_args[0])

    env.map_function_template(SimpleFunctionTemplate("make-int-hashtable", make_int_hashtable))

    def get_hash_value(template, evaluated_args):
        return evaluated_args[0].get_value(evaluated_args[1])

    env.map_function_template(SimpleFunctionTemplate("gethash", get_hash_value))

    def set_hash_value(template, evaluated_args):
        key = evaluated_args[1]
        if key.is_string():
            key = key.string()
            return evaluated_args[0].set_value(key, evaluated_args[2])
        else:
            key = key.int_value()
            return evaluated_args[0].set_value(key, evaluated_args[2])
    env.map_function_template(SimpleFunctionTemplate("defhash", set_hash_value))

    def has_key(template, evaluated_args):
        key = evaluated_args[1]
        if key.is_string():
            key = key.string()
            if evaluated_args[0].has_key(key):
                return evaluated_args[0].get_value(key)
            else:
                return NULL_VALUE
        else:
            key = key.int_value()
            if evaluated_args[0].has_key(key):
                return evaluated_args[0].get_value(key)
            else:
                return NULL_VALUE
    env.map_function_template(SimpleFunctionTemplate("contains-key", has_key))

    def get_hash_keys(template, evaluated_args):
        if evaluated_args[0].is_string_hashtable():
            keys = []
            for key in evaluated_args[0].map.keys():
                keys.append(LispTools.make_str(key))
            return LispTools.make_list(keys)

        if evaluated_args[0].is_int_hashtable():
            keys = []
            for key in evaluated_args[0].map.keys():
                keys.append(LispTools.make_integer(key))
            return LispTools.make_list(keys)

        raise Exception("Argument to get-hash-keys must be a hashtable")

    env.map_function_template(SimpleFunctionTemplate("get-hash-keys", get_hash_keys))

    def length(template, evaluated_args):
        if evaluated_args[0].is_list():
            return IntValue(evaluated_args[0].size())
        if evaluated_args[0].is_string():
            return IntValue(len(evaluated_args[0].string()))
        if evaluated_args[0].is_hashtable():
            return IntValue(evaluated_args[0].size())
        raise Exception("Argument to length must be a list, string, or hashtable")

    env.map_function_template(SimpleFunctionTemplate("length", length))

    def nth(template, evaluated_args):
        if not evaluated_args[0].is_list():
            raise Exception("First argument to nth must be a list")
        if not evaluated_args[1].is_integer():
            raise Exception("Second argument to nth must be an integer")
        index = evaluated_args[1].int_value()
        if index < 0 or index >= evaluated_args[0].size():
            raise Exception("Index out of range in nth")
        return evaluated_args[0].list()[index]

    env.map_function_template(SimpleFunctionTemplate("nth", nth))

    allow_insert_into_last_pos = True
    def set_nth(template, evaluated_args):
        if not evaluated_args[0].is_list():
            raise Exception("First argument to set-nth must be a list")
        if not evaluated_args[1].is_integer():
            raise Exception("Second argument to set-nth must be an integer")
        index = evaluated_args[1].int_value()


        if index < 0 or index > evaluated_args[0].size():
            raise Exception("Index out of range in set-nth")

        if index == evaluated_args[0].size():
            if not allow_insert_into_last_pos:
                raise Exception("Index out of range in set-nth")
            evaluated_args[0].list().append(evaluated_args[2])
        else:
            evaluated_args[0].list()[index] = evaluated_args[2]
        return evaluated_args[2]

    env.map_function_template(SimpleFunctionTemplate("set-nth", set_nth))

    def append_item(template, evaluated_args):
        if not evaluated_args[0].is_list():
            raise Exception("First argument to append-item must be a list")

        new = [l for l in evaluated_args[0].list()]
        new.append(evaluated_args[1])

        return LispTools.make_list(new)

    env.map_function_template(SimpleFunctionTemplate("append-item", append_item))

    def append_lists(template, evaluated_args):
        if not evaluated_args[0].is_list():
            raise Exception("First argument to append must be a list")

        if not evaluated_args[1].is_list():
            raise Exception("Second argument to append must be a list")

        return LispTools.make_list(evaluated_args[0].list() + evaluated_args[1].list())

    env.map_function_template(SimpleFunctionTemplate("append", append_lists))


    def unbind(template, evaluation_env):
        key = template.actual_arguments[0].string()

        target_env:Environment = evaluation_env
        search_env = evaluation_env
        while search_env is not None:
            if key in search_env.var_map:
                target_env = search_env
                break
            search_env = search_env.parent


        prior = target_env.has_value(key)
        if prior:
            target_env.unbind_value(key)
            return prior
        return NULL_VALUE

    env.map_function("unbind", unbind)

    def var_exists_p(template, evaluation_env:Environment):
        key = template.actual_arguments[0].string()

        if key in evaluation_env.var_map:
            return template.actual_arguments[0]
        if evaluation_env.parent:
            return var_exists_p(template, evaluation_env.parent)
        else:
            return NULL_VALUE

    env.map_function("var-exists-p", var_exists_p)

    return env



def add_arithmetic_functions(env):
    def add(template, evaluation_env):
        total = 0
        for arg in template.actual_arguments:
            total += arg.evaluate(evaluation_env).int_value()
        return IntValue(total)

    def subtract(template, evaluation_env):
        total = template.actual_arguments[0].evaluate(evaluation_env).int_value()
        for arg in template.actual_arguments[1:]:
            total -= arg.evaluate(evaluation_env).int_value()
        return IntValue(total)

    def multiply(template, evaluation_env):
        total = 1
        for arg in template.actual_arguments:
            total *= arg.evaluate(evaluation_env).int_value()
        return IntValue(total)

    def divide(template, evaluation_env):
        total = template.actual_arguments[0].evaluate(evaluation_env).int_value()
        for arg in template.actual_arguments[1:]:
            total /= arg.evaluate(evaluation_env).int_value()
        return IntValue(total)
    
    def to_integer(template, evaluated_args):
        return IntValue(evaluated_args[0].int_value())
    
    env.map_function_template(SimpleFunctionTemplate("integer", to_integer))
    
    def to_float(template, evaluated_args):
        return FloatValue(evaluated_args[0].float_value())
    
    env.map_function_template(SimpleFunctionTemplate("float", to_float))

    env.map_function("+", add)
    env.map_function("-", subtract)
    env.map_function("*", multiply)
    env.map_function("/", divide)


    return env



def check_numeric_args(evaluated_args):
    for arg in evaluated_args:
        if not arg.is_number():
            raise Exception("Arguments to arithmetic functions must be numbers")


def to_null(value, true_value = IntValue(1)):
    if value is None:
        return NULL_VALUE
    if value == False:
        return NULL_VALUE
    return true_value


def map_binding_list(binding_list, evaluation_env):
    if binding_list.is_list():
        bindings = binding_list.list()
        for binding in bindings:
            if binding.is_list():
                key = binding.list()[0].string()
                value = binding.list()[1].evaluate(evaluation_env)
                evaluation_env.map_value(key, value)
            else:
                key = binding.string()
                value = Environment.NULL_VALUE
                evaluation_env.map_value(key, value)
    return evaluation_env