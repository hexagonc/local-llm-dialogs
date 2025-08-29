import os

from .Environment import Environment
from .LispTools import LispTools
from .SimpleFunctionTemplate import SimpleFunctionTemplate
from .Value import NULL_VALUE
from .LispException import LispException
from LLMTools import read_prompt_file, write_string_to_file


CWD_VAR_NAME = "current-working-directory"
def add_filesystem_functions(env:Environment):

    env.map_value(CWD_VAR_NAME, LispTools.make_str(os.getcwd()))

    def get_current_directory(template, evaluated_args):
        return env.get_value(CWD_VAR_NAME)
    env.map_function_template(SimpleFunctionTemplate("get-cwd", get_current_directory))

    func_name = "cd"
    def set_current_directory(template, evaluated_args):
        if len(evaluated_args) == 1:
            target_directory = evaluated_args[0]

            target_directory = os.path.expanduser(target_directory)

            # Check if the new directory is valid
            if not os.path.isdir(target_directory):
                raise LispException(f"cd: no such file or directory: {target_directory}")

            os.chdir(target_directory)
            o = LispTools.make_str(os.getcwd())
            env.map_value(CWD_VAR_NAME, o)
            return o
        else:
            raise LispException(f"Incorrect argument count for {func_name}")

    env.map_function_template(SimpleFunctionTemplate(func_name, set_current_directory))

    func_name = "read-file"
    def read_file(template, evaluated_args):
        if len(evaluated_args) == 1:
            file_full_name = evaluated_args[0]

            # Check if the new directory is valid
            if not os.path.isfile(file_full_name):
                raise LispException(f"{func_name}: no such file: {file_full_name}")


            o = LispTools.make_str(read_prompt_file(file_full_name))
            return o
        else:
            raise LispException(f"Incorrect argument count for {func_name}")

    env.map_function_template(SimpleFunctionTemplate(func_name, read_file))

    func_name = "write-file"

    def write_file(template, evaluated_args):
        if len(evaluated_args) == 2:
            file_full_name = evaluated_args[0]

            # Check if the new directory is valid
            if not os.path.isfile(file_full_name):
                raise LispException(f"{func_name}: no such file: {file_full_name}")

            svalue = evaluated_args[1].string()
            write_string_to_file(svalue, file_full_name)
            return evaluated_args[1]
        else:
            raise LispException(f"Incorrect argument count for {func_name}")

    env.map_function_template(SimpleFunctionTemplate(func_name, write_file))

