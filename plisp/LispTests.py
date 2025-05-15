import unittest

from Environment import Environment
from Value import Value, ValueType
from LispTools import LispTools, add_arithmetic_functions, add_basic_functions
from SimpleFunctionTemplate import SimpleFunctionTemplate
from Parser import parse, NULL_VALUE


class TestLispFunctions(unittest.TestCase):
    def test_create_environment(self):
        env = Environment()

    def test_can_parse_numeric_literals(self):
        integer = 42
        nValue = LispTools.make_integer(integer)
        self.assertTrue(isinstance(nValue, Value))
        self.assertEqual(nValue.int_value(), integer)

    def test_can_create_list_type(self):
        raw = [LispTools.make_integer(i) for i in range(5)]
        listValue = LispTools.make_list(raw)
        self.assertTrue(isinstance(listValue, Value))
        self.assertEqual(listValue.get_type(), ValueType.LIST)
        values = listValue.list()
        self.assertEqual(len(values), 5)

    def test_can_create_string_literal(self):
        string = "hello"
        sValue = LispTools.make_str(string)
        self.assertTrue(isinstance(sValue, Value))
        self.assertEqual(sValue.get_type(), ValueType.STRING)
        self.assertEqual(sValue.string(), string)
        serialized = sValue.serialize()
        self.assertEqual( f"\"{string}\"", serialized)

    def test_can_create_symbol(self):
        symbol = "hello"
        sValue = LispTools.make_symbol(symbol)
        self.assertTrue(isinstance(sValue, Value))
        self.assertEqual(sValue.get_type(), ValueType.STRING)
        self.assertEqual(sValue.string(), symbol)
        serialized = sValue.serialize()
        self.assertEqual(serialized, symbol)

    def test_can_parse_primitive_types(self):
        env = Environment()

        exp = ""

        result = parse(exp)

        self.assertTrue(result is None)

        exp = "42"
        result = parse(exp)[0]
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.get_type(), ValueType.INTEGER)
        self.assertEqual(result.int_value(), 42)

    def test_can_parse_list(self):
        env = Environment()

        exp = "()"
        result = parse(exp)[0]
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.get_type(), ValueType.LIST)
        self.assertEqual(result.size(), 0)

        exp = "(42)"
        result = parse(exp)[0]
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.get_type(), ValueType.LIST)
        self.assertEqual(result.size(), 1)
        self.assertEqual(result.list()[0].int_value(), 42)

    def test_can_parse_function_call(self):
        env = Environment()

        exp = "(+ 1 2)"
        result = parse(exp)[0]
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.get_type(), ValueType.LIST)
        self.assertEqual(result.size(), 3)
        self.assertEqual(result.list()[0].string(), "+")
        self.assertEqual(result.list()[1].int_value(), 1)
        self.assertEqual(result.list()[2].int_value(), 2)
        self.assertTrue(result.list()[0].is_symbol)

    def test_can_can_parse_floats(self):
        env = Environment()

        exp = "3.14"
        result = parse(exp)[0]
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.get_type(), ValueType.FLOAT)
        self.assertEqual(result.float_value(), 3.14)

        exp = "(+ 3.14 3.14)"
        result = parse(exp)[0]
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.get_type(), ValueType.LIST)
        self.assertEqual(result.size(), 3)
        self.assertEqual(result.list()[0].string(), "+")
        self.assertEqual(result.list()[1].float_value(), 3.14)
        self.assertEqual(result.list()[2].float_value(), 3.14)

    def test_can_map_variables_into_environment(self):
        env = Environment()
        env.map_value("x", LispTools.make_integer(42))
        val = env.get_value("x")
        self.assertTrue(isinstance(val, Value))
        self.assertEqual(val.int_value(), 42)

        child = Environment(env)
        self.assertTrue(child.get_value("x") is not None)
        self.assertTrue(child.get_value("y") is None)

    def test_can_evaluate_values(self):
        env = Environment()
        env.map_value("x", LispTools.make_integer(42))
        result = env.evaluate(LispTools.make_symbol("x"))
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.int_value(), 42)


    def test_can_define_functions(self):
        env = Environment()
        add_arithmetic_functions(env)
        exp = parse("(+ 1 2)")[0]
        result = env.evaluate(exp)
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.int_value(), 3)

        def length(template, evaluation_env):
            argument = template.actual_arguments[0].evaluate(evaluation_env)
            return LispTools.make_integer(argument.size())

        env.map_function("length", length)
        exp = parse("(length (1 2 3 4 5))")[0]
        result = env.evaluate(exp)
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.int_value(), 5)

        def setq(template, evaluation_env):
            key = template.actual_arguments[0].string()
            value = template.actual_arguments[1].evaluate(evaluation_env)
            evaluation_env.map_value(key, value)
            return value

        env.map_function("setq", setq)

        def equals(template, evaluated_args):
            lvalue = evaluated_args[0]
            rvalue = evaluated_args[1]
            if lvalue.serialize() == rvalue.serialize():
                return lvalue
            else:
                return NULL_VALUE

        env.map_function_template(SimpleFunctionTemplate("=", equals))

        exp = parse("(setq x 42)")[0]
        result = env.evaluate(exp)
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.int_value(), 42)

        comp = parse("(= x 42)")[0]
        result = env.evaluate(comp)
        self.assertTrue(isinstance(result, Value))
        self.assertEqual(result.get_type(), ValueType.INTEGER)

        comp = parse("(= x 43)")[0]
        result = env.evaluate(comp)
        self.assertTrue(isinstance(result, Value))
        self.assertTrue(result)

    def test_can_define_conditional_and_looping_functions(self):
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)
        res = parse("(if (= 1 1) 42 43)")[0]
        result = env.evaluate(res)
        self.assertEqual(result.int_value(), 42)

        res = parse("(if (= 1 2) 42 43)")[0]
        result = env.evaluate(res)
        self.assertEqual(result.int_value(), 43)

        res = parse("(if (= 1 2) 42)")[0]
        result = env.evaluate(res)
        self.assertEqual(result.get_type(), ValueType.NULL)

        env.evaluate_exp("(setq x 10)")
        res = env.evaluate_exp("(for i 10 x (progn (print x) (setq x (* i 10))))")
        print(f"Final answer\: {res.int_value()}")
        self.assertEqual( 90, res.int_value())

        env.evaluate_exp("(setq y (mapcar x (1 2 3) (* 2 x)))")
        env.evaluate_exp("(print \"result is \" y)")


    def test_make_string_hashtable(self):
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)
        res = env.evaluate_exp("(make-string-hashtable)")
        self.assertTrue(isinstance(res, Value))
        self.assertTrue(res.get_type(), ValueType.STRING_HASHTABLE)

    def test_make_int_hashtable(self):
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)
        res = env.evaluate_exp("(make-int-hashtable)")
        self.assertTrue(isinstance(res, Value))
        self.assertTrue(res.get_type(), ValueType.INT_HASHTABLE)

    def test_get_int_hashtable_value(self):
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)
        res = env.evaluate_exp("(progn (setq map (make-int-hashtable)) (defhash map 12 90) (gethash map 12))")
        self.assertTrue(isinstance(res, Value))
        expected = LispTools.make_integer(90)
        self.assertTrue(res.get_type(), ValueType.INTEGER)
        self.assertTrue(expected.int_value() == res.int_value())



if __name__ == '__main__':
    unittest.main()