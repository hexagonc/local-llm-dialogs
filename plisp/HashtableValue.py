
from Value import Value, ValueType, StringValue, ListValue
from Value import Value, ValueType, ListValue, IntValue
from Environment import Environment
from plisp.Parser import NULL_VALUE


class HashtableValue(Value):
    def __init__(self, value_type):
        super().__init__(value_type)
        self.map = {}

    def get_value(self, key):
        if key in self.map:
            return self.map[key]
        else:
            return NULL_VALUE

    def set_value(self, key, value):
        self.map[key] = value
        return value

    def has_key(self, key):
        return key in self.map

    def size(self):
        return len(self.map)

class IntHashtableValue(HashtableValue):
    def __init__(self, item_spec):
        super().__init__(ValueType.INT_HASHTABLE)
        # item_spec is a list of lists. Each list has two elements: a key and a value.
        # The key is a string and the value is a Value object.
        for key_value_list_value in item_spec.list():
            key = key_value_list_value.list()[0].int_value()
            value = key_value_list_value.list()[1]
            self.map[key] = value

    def copy(self):
        items = [ListValue([IntValue(kv_pair[0]), kv_pair[1].copy()]) for kv_pair in self.map.items()]
        return IntHashtableValue(ListValue(items))

    def is_int_hashtable(self):
        return True

    def get_value(self, key):
        if key.is_integer() and  key.int_value() in self.map:
            return self.map[key.int_value()]
        else:
            return NULL_VALUE


class StringHashtableValue(HashtableValue):
    def __init__(self, item_spec):
        super().__init__(ValueType.STRING_HASHTABLE)
        # item_spec is a list of lists. Each list has two elements: a key and a value.
        # The key is a string and the value is a Value object.
        for key_value_list_value in item_spec.list():
            key = key_value_list_value.list()[0].string()
            value = key_value_list_value.list()[1]
            self.map[key] = value

    def copy(self):
        items = [ListValue([StringValue(kv_pair[0]), kv_pair[1].copy()]) for kv_pair in self.map.items()]
        return StringHashtableValue(ListValue(items))

    def is_string_hashtable(self):
        return True

    def get_value(self, key):
        if key.is_string() and key.string() in self.map:
            return self.map[key.string()]
        else:
            return NULL_VALUE