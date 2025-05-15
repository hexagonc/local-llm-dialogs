


class ValueType:
    STRING = 0
    INTEGER = 1
    FLOAT =2
    LIST = 3
    LAMBDA = 4
    NULL = 5
    STRING_HASHTABLE = 6
    INT_HASHTABLE =7
    OBJECT = 8


class Value:
    def __init__(self, type):
        self.type = type
        self.comma_delimited = False
        self.comma_list_delimited = False

    def __str__(self):
        if self.is_symbol():
            return self.string()
        else:
            return self.serialize()

    def is_key_name(self):
        return False

    def is_comma_delimited(self):
        return self.comma_delimited

    def is_comma_list_delimited(self):
        return self.comma_list_delimited

    def is_integer(self):
        return False

    def is_float(self):
        return False

    def is_list(self):
        return False

    def is_string_hashtable(self):
        return False

    def is_int_hashtable(self):
        return False

    def is_hashtable(self):
        return self.is_string_hashtable() or self.is_int_hashtable()

    def equals(self, other):
        return self.serialize() == other.serialize()

    def set_comma_list_delimited(self):
        self.comma_list_delimited = True

    def get_type(self):
        return self.type

    def int_value(self):
        return None

    def is_null(self):
        return False

    def is_number(self):
        return False

    def float_value(self):
        return None

    def size(self):
        return None

    def list(self):
        return None

    def set_comma_delimited(self):
        self.comma_delimited = True

    def string(self):
        return None

    def is_symbol(self):
        return False

    def is_string(self):
        return False

    def serialize(self):
        return "null"

    def copy(self):
        raise NotImplementedError("Copy not implemented")

    def evaluate(self, env):
        return self


class IntValue(Value):
    def __init__(self, value):
        super().__init__(ValueType.INTEGER)
        self.value = value

    def int_value(self):
        return self.value

    def float_value(self):
        return float(self.value)

    def serialize(self):
        return f"{self.value}"

    def is_integer(self):
        return True

    def copy(self):
        return IntValue(self.value)
    def is_number(self):
        return True


class FloatValue(Value):
    def __init__(self, value):
        super().__init__(ValueType.FLOAT)
        self.value = value

    def is_number(self):
        return True

    def is_float(self):
        return True

    def float_value(self):
        return self.value

    def int_value(self):
        return int(self.value)

    def serialize(self):
        return f"{self.value}"

    def copy(self):
        return FloatValue(self.value)

class ListValue(Value):
    def __init__(self, values):
        super().__init__(ValueType.LIST)
        self.values = values

    def list(self):
        return self.values

    def is_list(self):
        return True

    def size(self):
        return len(self.values)

    def copy(self):
        return ListValue([v.copy() for v in self.values])

    def serialize(self):
        return f"({', '.join([v.serialize() for v in self.values])})"

    def evaluate(self, env):
        if self.size() == 0:
            return self
        if self.values[0].is_symbol() and not self.values[0].is_key_name():
            function = env.get_function(self.values[0].string())
            if function is not None:
                function.set_actual_arguments(self.values[1:])
                return function.evaluate(env)
            else:
                raise Exception(f"Function {self.values[0].string()} not found")
        return self




class StringValue(Value):
    def __init__(self, value, is_symbol=False):
        super().__init__(ValueType.STRING)
        self.value = value
        self.symbol_p = is_symbol

    def is_key_name(self):
        return self.is_symbol() and self.value.startswith(":")

    def is_symbol(self):
        return self.symbol_p

    def is_string(self):
        return True

    def string(self):
        return self.value

    def copy(self):
        return StringValue(self.value, self.is_symbol())

    def serialize(self):
        if self.is_symbol():
            return self.value
        else:
            return f"\"{self.value}\""


    def evaluate(self, env):
        if self.is_symbol():
            res = env.get_value(self.value)
            if res is None:
                return NULL_VALUE
            else:
                return res
        return self

NULL_LITERAL = "F"

class NullValue(Value):
    def __init__(self):
        super().__init__(ValueType.NULL)

    def is_null(self):
        return True

    def serialize(self):
        return NULL_LITERAL

    def is_symbol(self):
        return True

    def copy(self):
        return self


NULL_VALUE = NullValue()