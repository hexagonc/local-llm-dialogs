import unittest
from plisp.Environment import Environment
from plisp.LispTools import add_basic_functions
from plisp.LispTools import add_arithmetic_functions
from plisp.FileTools import add_filesystem_functions

class MyTestCase(unittest.TestCase):
    def test_something(self):
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)
        add_filesystem_functions(env)

        #self.assertEqual(True, False)  # add assertion here

if __name__ == '__main__':
    unittest.main()
