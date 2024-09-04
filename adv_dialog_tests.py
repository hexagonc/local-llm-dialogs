import unittest
import os
from LLMDialogController import LLMDialogController
from LLMTools import read_text, apply_custom_delimiter, dialog_token_size, PATH_SEP


HOME = "/Users/my_computer"
class MyTestCase(unittest.TestCase):

    def test_can_access_ollama_model(self):
        from LLMTools import do_one_shot_llm_query

        url = "http://localhost:11434/v1/"
        api_key = "ollama"
        query = "Show the code for displaying the current date and time from a bash terminal.  Surround this command using delimiters like this: /**{terminal command}**/"
        model = "llama3.1:8b-instruct-fp16"
        name = "model"
        resp = do_one_shot_llm_query(query = query, url=url, api_key=api_key, llm_name=model)
        print(resp)


    def test_can_replay_history_n_steps(self):
        initial_dialog_file = "./dialog_project_cloning_bash.json"
        dialog = LLMDialogController(initial_dialog_file = initial_dialog_file)
        resp = dialog.chat("system: replay 4")
        print(resp)

    def test_can_load_gemma2_model(self):
        dialog = LLMDialogController()
        dialog.chat("system: use model gemma2", contWithStd=False)
        resp = dialog.chat("what did the chicken cross the road?", contWithStd=False)
        print(resp)

    # this is optional
    def test_can_use_high_precision_llama_model(self):
        from LLMTools import match_pattern
        pattern = "use model name {model_name}"
        input = "use model llama3-precision"
        map = match_pattern(pattern, input, llm_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")

        self.assertTrue("model_name" in map)
        self.assertEqual("llama3-precision", map["model_name"])
        print(map)


    def test_can_create_dialog_controller(self):
        dialog = LLMDialogController()

    def test_can_impersonate_a_user_than_save(self):
        d1 = "user: In the following set of instructions and examples, I will show you how to accomplish a goal by executing a sequence of shell commands on a computer.  I will give you a task to complete and you will analyze that task and break it up into a sequence of steps in the form of shell commands that I will execute on your behalf to complete that task."
        d2 = "assistant: understood"
        d3 = "user: all shell commands that you want me to execute directly must be delimited by /* */ since it will be intermingled with java code and so needs to appear like a comment in order to avoid being processed by the java compiler"
        d4 = "assistant: understood"
        d6 = f"""user: for example, if I find the following code in your response: /*ls ~*/ then you can assume that I will try to run 'ls ~' as soon as possible within my command shell.  Assume my shell has an initial working directory of '{HOME}/development/projects/desktop_llms/test/playground'.  Upon executing that command in the shell, I will take note of what gets written to standard output and pass that along to you.  For example, if running 'ls ~' in the shell returns:
```
Desktop
Documents
Downloads
Google Drive
Library
Movies
Music
Pictures
Public
development
```
then the next input from me will be:
```
stdout:
Desktop
Documents
Downloads
Google Drive
Library
Movies
Music
Pictures
Public
development 
``` 
Notice that my response will be prefixed by 'stdout:' when I am relaying to you the result of executing a shell command.  If executing the command
you specify results in an error from the shell, as would be the case if you responded with /*ls ~/junk*/ then my response will be prefixed by 'stderr:' like this:
```
stderr:
ls: {HOME}/junk: No such file or directory
```
At this point, you can either correct the original command or you can ask me for help and we can troubleshoot the problem together.  
In some cases, I may ask you to do something that is impossible for you to accomplish with shell commands, in which case
you can ask me for help or you can declare that the task is impossible to achieve.
"""
        d7 = "assistant: understood"
        d8 = f"user: suppose I asked: \"is 'temp' a child directory of {HOME}/development/projects/desktop_llms/test/playground?\"  How would you respond?"
        d9 = f"assistant: first execute /*ls {HOME}/development/projects/desktop_llms/test/playground*/.  Doing this should show the files in that directory"
        d10 = """user:stdout:
base_automata_path
default_dialog.json
dialogs
knowledgebase
my_likes_and_dislikes.json
"""
        d11 = f"assistant: No, 'temp' is not a child directory of {HOME}/development/projects/desktop_llms/test/playground"
        dialog = LLMDialogController()
        dialog_pre_history = [d1, d2, d3, d4, d6, d7, d8, d9, d10, d11]
        history = dialog.current_dialog.dialog_history
        initial_length = len(history)
        for user_input in dialog_pre_history:
            dialog.chat(user_input, contWithStd = False)
            self.assertTrue(len(history) == initial_length+1)
            initial_length = len(history)

        self.assertTrue(len(history) == (len(dialog_pre_history) +1))
        resp = dialog.chat("Now how do I create a folder called 'temp' in the directory ~ and then create a file called 'default.json' in it?  Provide these commands one at a time as I confirm there execution", contWithStd = False)
        print(resp)

    def test_can_create_new_dialog_branch_file(self):
        d1 = "user: In the following set of instructions and examples, I will show you how to accomplish a goal by executing a sequence of shell commands on a computer.  I will give you a task to complete and you will analyze that task and break it up into a sequence of steps in the form of shell commands that I will execute on your behalf to complete that task."
        d2 = "assistant: understood"
        d3 = "user: all shell commands that you want me to execute directly must be delimited by /* */ since it will be intermingled with java code and so needs to appear like a comment in order to avoid being processed by the java compiler"
        d4 = "assistant: understood"
        d6 = f"""user: for example, if I find the follow code in your response: /*ls ~*/ then you can assume that I will try to run 'ls ~' as soon as possible within my command shell.  Assume my shell has an initial working directory of '{HOME}/development/projects/desktop_llms/test/playground'.  Upon executing that command in the shell, I will take note of what gets written to standard output and pass that along to you.  For example, if running 'ls ~' in the shell returns:
        ```
        Desktop
        Documents
        Downloads
        Google Drive
        Library
        Movies
        Music
        Pictures
        Public
        development
        ```
        then the next input from me will be:
        ```
        stdout:
        Desktop
        Documents
        Downloads
        Google Drive
        Library
        Movies
        Music
        Pictures
        Public
        development 
        ``` 
        Notice that my response will be prefixed by 'stdout:' when I am relaying to you the result of executing a shell command.  If executing the command
        you specify results in an error from the shell, as would be the case if you responded with /*ls ~/junk*/ then my response will be prefixed by 'stderr:' like this:
        ```
        stderr:
        ls: {HOME}/junk: No such file or directory
        ```
        At this point, you can either correct the original command or you can ask me for help and we can troubleshoot the problem together.  
        In some cases, I may ask you to do something that is impossible for you to accomplish with shell commands, in which case
        you can ask me for help or you can declare that the task is impossible to achieve.  Some commands may not have either stdout or stderr in which case, I 
        will simply respond with the return code, which will be something like: 'returncode: 0' in the case of success.  In the cases where the command fails but 
        there is nothing in stderr, I will simply reply with a non-zero return code, such as 'returncode: 1'.
        """
        d7 = "assistant: understood"
        d8 = f"user: suppose I asked: \"is 'temp' a child directory of {HOME}/development/projects/desktop_llms/test/playground?\"  How would you respond?"
        d9 = f"assistant: first execute /*ls {HOME}/development/projects/desktop_llms/test/playground*/.  Doing this should show the files in that directory"
        d10 = """user:stdout:
        base_automata_path
        default_dialog.json
        dialogs
        knowledgebase
        my_likes_and_dislikes.json
        """
        d11 = f"assistant: No, 'temp' is not a child directory of {HOME}/development/projects/desktop_llms/test/playground"
        dialog = LLMDialogController()
        dialog_pre_history = [d1, d2, d3, d4, d6, d7, d8, d9, d10, d11]
        for user_input in dialog_pre_history:
            dialog.chat(user_input, contWithStd=False)

        new_file_name = "filesystem_actions.json"

        resp = dialog.chat(f"system: create branch {new_file_name}", contWithStd=False)
        print(resp)
        expected_filename = f"{dialog.dialog_index_path}{PATH_SEP}{new_file_name}"
        self.assertTrue(os.path.isfile(expected_filename))

    def test_can_import_dialog_branch_file(self):
        dialog = LLMDialogController()
        existing_file_name = "filesystem_actions.json"

        resp = dialog.chat(f"system: import {existing_file_name}", contWithStd=False)
        print(resp)
        expected_filename = f"{dialog.dialog_index_path}{PATH_SEP}{existing_file_name}"
        self.assertTrue(os.path.isfile(expected_filename))
        resp = dialog.chat("Now how do I create a folder called 'temp' in the directory ~ and then create a file called 'default.json' in it?  Provide these commands one at a time as I confirm there execution", contWithStd = False)
        print(resp)
        resp = dialog.chat(
            "returncode: 0",
            contWithStd=False)
        print(resp)

    def test_can_pop_last_user_action_response(self):
        dialog = LLMDialogController()
        existing_file_name = "filesystem_actions.json"

        resp = dialog.chat(f"system: import {existing_file_name}", contWithStd=False)
        print(resp)
        expected_filename = f"{dialog.dialog_index_path}{PATH_SEP}{existing_file_name}"
        self.assertTrue(os.path.isfile(expected_filename))
        resp = dialog.chat("user: to_do: add pickup grapes", contWithStd = False)
        print(resp)
        resp = dialog.chat("assistant: to_do: thanks, I have added grapes to your to do list", contWithStd=False)
        print(resp)
        dialog_length = len(dialog.current_dialog.dialog_history)
        resp = dialog.chat("system: pop", contWithStd = False)
        print(resp)
        new_dialog_length = len(dialog.current_dialog.dialog_history)
        self.assertTrue(dialog_length-2 == new_dialog_length)


    def test_invalid_dialog_file(self):
        print()
        dialog = LLMDialogController()
        existing_file_name = "filesdystem_actions.json"

        resp = dialog.chat(f"system: import {existing_file_name}", contWithStd=False)

        resp = dialog.chat("list all files in user home directory", contWithStd = False)
        dialog_len = len(dialog.current_dialog.dialog_history)
        resp = dialog.chat("system: evaluate shell command", contWithStd = False)
        self.assertEqual(dialog_len , len(dialog.current_dialog.dialog_history))


    def test_can_match_detailed_patterns_with_llms(self):
        from LLMTools import match_pattern
        pattern = "use model name {model_name}"
        input = "use model llama3"
        map = match_pattern(pattern, input)
        self.assertTrue("model_name" in map)
        self.assertEqual("llama3", map["model_name"])
        print(map)



if __name__ == '__main__':
    unittest.main()
