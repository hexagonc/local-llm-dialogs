import os
import unittest
from plisp.Environment import Environment
from plisp.LispTools import add_basic_functions
from plisp.LispTools import add_arithmetic_functions
from plisp.FileTools import add_filesystem_functions
from plisp.Value import Value

class MyTestCase(unittest.TestCase):

    def setUp(self):
        print("\n")

    def test_something(self):
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)
        add_filesystem_functions(env)

        #self.assertEqual(True, False)  # add assertion here

    def test_can_get_current_working_directory(self):
        expected_value = os.getcwd()
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)
        add_filesystem_functions(env)

        res:Value = env.evaluate_exp("(get-cwd)")
        self.assertTrue(res.string(), expected_value)
        print(f"current working directory: {res.string()}")

    def test_can_change_current_working_directory(self):
        expected_value = os.getcwd()
        env = Environment()
        add_arithmetic_functions(env)
        add_basic_functions(env)
        add_filesystem_functions(env)

        env.map_value("dir", "plisp")
        res:Value = env.evaluate_exp("(cd dir)")
        print(f"New directory: {res}")



    def test_can_create_lisp_dialog_activity_files(self):
        activity_name = "return lisp code that accomplishes a task"
        dir = os.getcwd()
        activity_name_base_file_name = "get_lisp.txt"
        activity_file_abs_path = f"{dir}/{activity_name_base_file_name}"




if __name__ == '__main__':
    unittest.main()
