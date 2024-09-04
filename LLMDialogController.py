import os.path
import threading
import queue

from LinearLLMDialog import LinearLLMDialog
from SpeechHashMap import SpeechHashMap
from LLMTools import deserialize_from_file, serialize_to_file, get_short_filename, \
    extract_files_or_folders_from_user_input, LM_STUDIO_API_URL, LM_STUDIO_API_KEY
from typing import Optional
from LLMTools import user_prompt_segment, assistant_prompt_segment, get_delimited_text
from LLMPatternMatcher import LLMPatternMatcher
from LLMTools import run_shell_command, read_text, apply_custom_delimiter, write_string_to_file, dialog_token_size, PATH_SEP


DIALOG_INDEX_FILE_SHORT = "dialog_index.json"
DEFAULT_DIALOG_FILE_SHORT = "default.json"

SYSTEM_COMMAND_PREFIX = "system: "


class LLMDialogController:
    def __init__(self, initial_dialog_file = None, dialog_index_path = None, display_responses = None):
        self.last_shell_command = None

        if dialog_index_path is None:
            dialog_index_path = os.getcwd()
        self.dialog_index_path = dialog_index_path
        if initial_dialog_file is None:
            initial_dialog_file = f"{self.dialog_index_path}{PATH_SEP}{DEFAULT_DIALOG_FILE_SHORT}"
        self.current_dialog_file = initial_dialog_file
        self.dialog_index_file_name = f"{dialog_index_path}{PATH_SEP}{DIALOG_INDEX_FILE_SHORT}"

        if os.path.isfile(self.dialog_index_file_name):
            data = deserialize_from_file(self.dialog_index_file_name)
            self.dialog_index_map = SpeechHashMap(data)
            self.dialog_index_map.set(DEFAULT_DIALOG_FILE_SHORT,
                                      f"{self.dialog_index_path}{PATH_SEP}{DEFAULT_DIALOG_FILE_SHORT}")
        else:
            self.dialog_index_map = SpeechHashMap()
            self.dialog_index_map.set(DEFAULT_DIALOG_FILE_SHORT, f"{self.dialog_index_path}{PATH_SEP}{DEFAULT_DIALOG_FILE_SHORT}")
            self.saveIndex()

        self.current_dialog = LinearLLMDialog()
        if os.path.exists(self.current_dialog_file):
            self.current_dialog.loadDialog(self.current_dialog_file)
        self.dialog_is_being_edited = False
        if display_responses is None:
            display_responses = True
        self.display_responses = display_responses

        self.command_map = SpeechHashMap()

        replay_key = "replay {number}"
        self.command_map.set(replay_key, self.tryReplayingDialogHistory)

        self.file_name_key = "file_name"
        switch_branch_command_1 = f"switch to branch {self.file_name_key}"
        switch_branch_command_2 = f"go to branch {self.file_name_key}"
        switch_branch_command_3 = f"create branch {self.file_name_key}"

        set_model = "use model name"
        self.command_map.set(set_model, self.set_default_llm_name)

        use_branch_command_1 = f"use {self.file_name_key}"
        use_branch_command_2 = f"import {self.file_name_key}"

        eval_1 = f"evaluate shell command"
        eval_2 = f"run shell command"

        pop_dialog_1 = "pop {num_steps}"

        self.context_file_name = None

        show_content_1 = f"show assistant content"
        self.command_map.set(show_content_1, self.tryShowAssistantContentCommand)

        set_introduce_file = "introduce the file: file_name"
        self.command_map.set(set_introduce_file, self.tryIntroduceNewFileCommand)

        set_write_content_to_context_file = "write assistant content to file_name"
        self.command_map.set(set_write_content_to_context_file, self.tryWritingAssistantContentToFile)

        self.command_map.set(pop_dialog_1, self.tryPoppingDialog)

        self.command_map.set(switch_branch_command_1, self.trySwitchingBranches)
        self.command_map.set(switch_branch_command_2, self.trySwitchingBranches)
        self.command_map.set(switch_branch_command_3, self.trySwitchingBranches)

        self.command_map.set(use_branch_command_1, self.tryUsingReadOnlyBranch)
        self.command_map.set(use_branch_command_2, self.tryUsingReadOnlyBranch)

        self.command_map.set(eval_1, self.tryEvaluateLastShellCommand)
        self.command_map.set(eval_2, self.tryEvaluateLastShellCommand)

        self.CONTENT_START_DELIMITER = "```"
        self.CONTENT_END_DELIMITER = "```"
        self.system_content_response = None
        self.user_content_file_name = None
        self.user_contents = None

        from LLMTools import LLAMA_LLM_NAME, LM_STUDIO_API_URL, LM_STUDIO_API_KEY
        from LLMTools import OPENAI_GPT4o, OPENAI_API_URL, OPENAI_API_KEY
        self.model_config_map = {"llama3": ("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", LM_STUDIO_API_URL, LM_STUDIO_API_KEY),
                                 "gemma2": ("bartowski/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q6_K-Q8.gguf", LM_STUDIO_API_URL, LM_STUDIO_API_KEY),
                                 "meta-llama3.1": ("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", LM_STUDIO_API_URL, LM_STUDIO_API_KEY),
                                 "llama3.1-storm": ("akjindal53244/Llama-3.1-Storm-8B-GGUF/Llama-3.1-Storm-8B.Q8_0.gguf", LM_STUDIO_API_URL, LM_STUDIO_API_KEY),
                                 "openai": (OPENAI_GPT4o, OPENAI_API_URL, OPENAI_API_KEY)
                                 }



    def set_default_llm_name(self, user_input):
        from LLMTools import match_pattern
        if user_input.startswith("use"):
            pattern = "use model name {model_name}"
            map = match_pattern(pattern, user_input)
            if "model_name" in map:
                name_key = map["model_name"]
                if name_key in self.model_config_map:
                    model_name, url, api_key = self.model_config_map[name_key]
                    self.current_dialog.model_name = model_name
                    self.current_dialog.model_url = url
                    self.current_dialog.model_api_key = api_key
                    return False, f"Changes default model to: {model_name}", False
                else:
                    explicit_model = map["model_name"]
                    self.current_dialog.model_name = explicit_model
                    self.current_dialog.model_url = LM_STUDIO_API_URL
                    self.current_dialog.model_api_key = LM_STUDIO_API_KEY
                    return False, f"Changed default model to: {explicit_model}", False
            else:
                return False, None, True
        else:
            return False, user_input, True

    def get_raw_text_from_assistant_response(self, response):
        text_vals = get_delimited_text(response, self.CONTENT_START_DELIMITER, self.CONTENT_END_DELIMITER)
        if text_vals and len(text_vals) > 0:
            self.system_content_response = text_vals
            return self.system_content_response
        return None

    def tryShowAssistantContentCommand(self, command:str):
        if command.startswith("show"):
            if self.system_content_response and len(self.system_content_response)> 0:
                return False, f"\n{self.system_content_response[0][0]}", False
            else:
                return True, "No assistant content to show", False
        else:
            return False, None, True

    def tryIntroduceNewFileCommand(self, command: str):
        if command.startswith("introduce"):
            user_content_file_name = extract_files_or_folders_from_user_input(command, files=True, folders=False)

            if user_content_file_name is None:
                return True, "No file to introduce the assistant to", False
            start_del, stop_del = "'''", "'''"

            try:
                contents = read_text(user_content_file_name)
                self.user_content_file_name = user_content_file_name
                self.user_contents = contents
            except Exception as e:
                return True, f"Error opening file: {user_content_file_name}", False
            self.displaySystemOutput("Communicating to assistant on behalf of user...")
            file_content = apply_custom_delimiter(contents, start_del, stop_del)
            self.CONTENT_END_DELIMITER, self.CONTENT_START_DELIMITER = (start_del, stop_del)
            user_command = f"Consider the file \"{user_content_file_name}\".  Using start and end delimiters: {start_del} and {stop_del}\n the contents of that file is: {file_content}"
            self.displayUserInput(user_command)

            resp = self.current_dialog.chat(user_command,temp=0.0)
            self.displayAssistantOutput(resp)
            return False, None, False
        else:
            return False, None, True




    def tryWritingAssistantContentToFile(self, command):
        if command.startswith("write"):
            content_output_file = extract_files_or_folders_from_user_input(command, files = True, folders=False)

            if content_output_file is None:
                return True, "No file to write to", False

            contents = self.system_content_response
            if contents and len(contents) > 0:
                try:
                    write_string_to_file(contents[0][0], content_output_file)
                    return False, f"Saved assistant content to: {content_output_file}", False
                except Exception as e:
                    return True, f"Failed to save file: {content_output_file}\n{e}", False
            else:
                resp = f"No assistant content to write"
                return True, resp, False
        else:
            return False, None, True

    def saveIndex(self):
        serialize_to_file(self.dialog_index_map.data_dict, self.dialog_index_file_name)

    def tryPreprocessUserInput(self, user_input):
        """
        Return value is a tuple ({error:Bool}, {user_input_to_pass_on:Optional[str]}, {should_continue:Bool})
        """
        top_command_lambdas = [self.tryParseUserImpersonation, self.tryParseAssistantImpersonation, self.tryParseSystemCommand]

        try:
            for top_handler in top_command_lambdas:
                was_error, user_input, cont = top_handler(user_input)
                if not cont:
                    return was_error, user_input, cont
        except Exception as e:
            return True, f"Error executing command: {user_input}\n\n{e}", False
        return False, user_input, True

    def tryReplayingDialogHistory(self, user_input:str):
        if user_input.startswith("replay"):
            pattern = "replay {num_steps}"
            matcher = LLMPatternMatcher()
            o = matcher.extractFields(user_input, pattern)
            if o is None:
                self.displaySystemOutput("Replayed the previous user and assistant response.")
                tete_a_tete = self.current_dialog.dialog_history[-2:]
                out = self.displayDialogSequence(tete_a_tete)
                return False, f"\n{out}", False
            elif "num_steps" in o:
                steps = o["num_steps"]
                if steps == "replay":
                    # this will happen if the number isn't provided
                    self.displaySystemOutput("Replayed the previous user and assistant response.")
                    tete_a_tete = self.current_dialog.dialog_history[-2:]
                    out = self.displayDialogSequence(tete_a_tete)
                    return False, f"\n{out}", False
                try:
                    num_steps = int(steps)
                    if num_steps > 1:
                        self.displaySystemOutput(f"Replayed {steps} previous user or assistant responses.")
                        tete_a_tete = self.current_dialog.dialog_history[-num_steps:]
                        out = self.displayDialogSequence(tete_a_tete)
                        return False, f"\n{out}", False
                except Exception as e:
                    return True, f"could not understand replay command: {user_input}", False

            return True, f"could not understand replay command: {user_input}", False
        return False, user_input, True


    def tryPoppingDialog(self, user_input:str):
        if user_input.startswith("pop"):
            pattern = "pop {num_steps}"
            matcher = LLMPatternMatcher()
            o = matcher.extractFields(user_input, pattern)
            if o is None:
                self.displaySystemOutput("Popped previous user and assistant response.")
                self.current_dialog.trimDialog(2)
                self.displayContext(4)
                return False, None, False
            elif "num_steps" in o:
                steps = o["num_steps"]
                if steps == "pop":
                    # this will happen if the number isn't provided
                    self.displaySystemOutput("Popped previous user and assistant response.")
                    self.current_dialog.trimDialog(2)
                    self.displayContext(4)
                    return False, None, False
                if isinstance(type(steps), (int, float)) and steps > 1:
                    self.displaySystemOutput(f"Popped {steps} previous user and assistant responses.")
                    self.current_dialog.trimDialog(steps)
                    self.displayContext(2*steps)
                    return False, None, False
            return True, f"could not understand pop command: {user_input}", False
        return False, user_input, True

    def executeLastShellCommand(self, command):
        try:
            self.last_shell_command = None
            system_response_to_assistant = None

            return_code, stdout, stderr = run_shell_command(command)
            if len(stderr) == 0:
                if len(stdout) > 0:
                    system_response_to_assistant = f"stdout:\n{stdout}"
                else:
                    system_response_to_assistant = f"returncode:{return_code}"
            else:
                system_response_to_assistant = f"stderr:{stderr}"
            return system_response_to_assistant
        except Exception as e:
            self.last_shell_command = None
            self.displaySystemErrorOutput(f"There was a system error executing '{command}':\n{e}")
            return None

    def tryEvaluateLastShellCommand(self, user_input:str):
        if self.last_shell_command and len(self.last_shell_command) > 0:
            from LLMTools import match_pattern
            map = match_pattern("run shell command ({index})", user_input)
            if "index" in map:
                countStr = map["index"]
                count = int(countStr)
                if count <= len(self.last_shell_command) and count > 0:
                    command = self.last_shell_command[count-1][0]
                else:
                    return True, f"Invalid command index: {countStr}", False
            else:
                command = self.last_shell_command[-1][0]

            response_for_assistant = self.executeLastShellCommand(command)
            if response_for_assistant:
                self.displayUserInput(response_for_assistant)
                resp = self.current_dialog.chat(response_for_assistant)
                self.parse_shell_commands(resp)
                self.displayAssistantOutput(resp)
                return False, None, False
            else:
                return True, None, False
        return True, "No command to run", False


    def tryParseSystemCommand(self, user_input:str):
        system_command_prefix = "system:"
        system_command = self.parsePrefix(system_command_prefix, user_input)
        if system_command:
            return self.processSystemCommand(system_command.strip())
        else:
            return False, user_input, True


    def parsePrefix(self, prefix:str, input:str) -> Optional[str]:
        """
        Returns the remaining string if it is prefixed by [prefix] otherwise returns None
        """
        plength = len(prefix)
        if input.startswith(prefix):
            return input[plength:]
        else:
            return None

    def tryUsingReadOnlyBranch(self, command):
        new_dialog_file = extract_files_or_folders_from_user_input(command)
        if new_dialog_file is None:
            return True, "No file to import", False
        base_dialog_path = os.path.dirname(new_dialog_file)
        dialog_name = get_short_filename(new_dialog_file, ignore_ext = False)
        if len(base_dialog_path) == 0:
            base_dialog_path = self.dialog_index_path
            new_dialog_file = f"{base_dialog_path}{PATH_SEP}{dialog_name}"

        if os.path.exists(new_dialog_file):
            self.current_dialog.loadDialog(new_dialog_file)
            self.dialog_is_being_edited = False
            self.current_dialog_file = new_dialog_file

            self.dialog_index_map.set(dialog_name, new_dialog_file)
            self.saveIndex()
            resp = f"Using the dialog file {dialog_name}"
            return False, resp, False
        else:
            resp = f"Can't use non-existent dialog file: {dialog_name}"
            return True, resp, False

    def trySwitchingBranches(self, command):
        switch_branch_command_3 = "create branch {file_name}"
        matcher = LLMPatternMatcher()
        pmap = matcher.extractFields(switch_branch_command_3, command)
        if "file_name" in pmap:
            new_dialog_file = pmap["file_name"]
            if new_dialog_file is None:
                return True, "No file to switch to", False
            base_dialog_path = os.path.dirname(new_dialog_file)
            dialog_name = get_short_filename(new_dialog_file, ignore_ext = False)
            if len(base_dialog_path) == 0:
                base_dialog_path = self.dialog_index_path
                new_dialog_file = f"{base_dialog_path}{PATH_SEP}{dialog_name}"
            existed = os.path.exists(new_dialog_file)
            self.current_dialog.startDialogBranchRecording(new_dialog_file)
            self.dialog_is_being_edited = True
            self.current_dialog_file = new_dialog_file
            self.dialog_index_map.set(dialog_name, new_dialog_file)
            self.saveIndex()
            if existed:
                resp = f"Switched to dialog: {dialog_name}"
            else:
                resp = f"Started new dialog: {dialog_name}"
            return False, resp, False
        else:
            return False, None, True

    def processSystemCommand(self, command):
        matching_command_handlers= self.command_map.find(command, 2)

        if matching_command_handlers:
            for key in matching_command_handlers:
                handler_lambda = self.command_map.get(key)
                error, response, cont = handler_lambda(command)
                if not cont:
                    return error, response, cont
        return True, f"unrecognized command: {command}", False

    def tryParseUserImpersonation(self, user_input:str):
        imp_user_command_prefix = "user:"
        command = self.parsePrefix(imp_user_command_prefix, user_input)
        if command:
            self.current_dialog.dialog_history.append(user_prompt_segment(command))
            return False, user_input, False
        else:
            return False, user_input, True

    def tryParseAssistantImpersonation(self, user_input:str):
        imp_assistant_command_prefix = "assistant:"
        command = self.parsePrefix(imp_assistant_command_prefix, user_input)
        if command:
            self.parse_shell_commands(command)
            self.current_dialog.dialog_history.append(assistant_prompt_segment(command))
            return False, user_input, False
        else:
            return False, user_input, True

    def parse_shell_commands(self, assistant_response):
        self.get_raw_text_from_assistant_response(assistant_response)
        expressions = get_delimited_text(assistant_response, "/*", "*/")
        if len(expressions) > 0:
            self.last_shell_command = expressions
            return self.last_shell_command
        else:
            return None


    def injectDialog(self, user, assistant):
        self.tryParseUserImpersonation(user)
        self.tryParseAssistantImpersonation(assistant)
        self.displayUserInput(user)
        self.displayAssistantOutput(assistant)

    def get_user_input(self, prompt = None, input_queue:queue.Queue = None):
        if input_queue:
            return input_queue.get()
        else:
            if prompt is None:
                prompt = ""
            return input(prompt)

    def chat(self, user_input:str = None, contWithStd:bool = None, async_task = None, user_input_queue = None):
        task = None
        message_queue = queue.Queue()
        async_response_queue = queue.Queue()
        data = {"inject-dialog":lambda user, assistant: self.injectDialog(user, assistant)}

        if async_task:
            task = threading.Thread(target=async_task, args=(data,))
            task.start()
        try:
            if contWithStd is None:
                contWithStd = False
            dialog_name = get_short_filename(self.current_dialog_file)

            if user_input is None or len(user_input.strip()) == 0:
                if contWithStd:
                    if self.dialog_is_being_edited:
                        input_prompt = f"{dialog_name}*: "
                    else:
                        input_prompt = f"{dialog_name}:"
                    user_input = self.get_user_input(input_prompt, user_input_queue)
                    self.displayUserInput(user_input)
                else:
                    return None
            else:
                self.displayUserInput(user_input)
            if not async_response_queue.empty():
                async_response = async_response_queue.get()
                async_user_input = f"update: {async_response}"
                self.displaySystemOutput()
                resp = self.current_dialog.chat(async_user_input, temp=0.0)
            while len(user_input.strip()) > 0:
                was_error, processed_input, pass_along_to_assistant = self.tryPreprocessUserInput(user_input)
                if pass_along_to_assistant:
                    original = self.current_dialog.dialog_history.copy()
                    try:
                        resp = self.current_dialog.chat(processed_input, temp=0.0)
                        self.parse_shell_commands(resp)
                        self.displayAssistantOutput(resp)
                    except Exception as e:
                        self.current_dialog.dialog_history = original
                        self.displaySystemErrorOutput(f"Error processing chat respnse: {e}")
                    if not contWithStd:
                        return resp
                else:
                    resp = processed_input
                    if was_error:
                        if processed_input:
                            self.displaySystemErrorOutput(processed_input)
                        else:
                            resp = f"failed to process command: {user_input}"
                            self.displaySystemErrorOutput(resp)
                    else:
                        if processed_input:
                            self.displaySystemOutput(resp)
                        else:
                            resp = f"Processed command: {user_input}"
                            self.displaySystemOutput(resp)
                    if not contWithStd:
                        return resp
                dialog_name = get_short_filename(self.current_dialog_file)

                if self.dialog_is_being_edited:
                    input_prompt = f"{dialog_name}*: "
                else:
                    input_prompt = f"{dialog_name}:"
                user_input = self.get_user_input(input_prompt, user_input_queue)
                self.displayUserInput(user_input)
        finally:
            if task:
                message_queue.put({})
                try:
                    task.join()
                except Exception as ee:
                    pass

    def displayUserInput(self, input):
        ret = ""
        if self.display_responses:
            ret = f"\033[34muser: \033[0m{input}\n"
            print(ret)
        return ret

    def displayAssistantOutput(self, output):
        ret = ""
        if self.display_responses:
            ret = f"\033[33massistant: \033[0m{output}\n"
            print(ret)
        return ret


    def displaySystemOutput(self, output):
        ret = ""
        if self.display_responses:
            ret = f"\033[32msystem: \033[0m{output}\n"
            print(ret)
        return ret

    def displaySystemErrorOutput(self, output):
        ret = ""
        if self.display_responses:
            ret = f"\033[31msystem: \033[0m{output}\n"
            print(ret)
        return ret

    def displayDialogSequence(self, dialog_sequence):
        out = []
        for seg in dialog_sequence:
            role = seg["role"]
            resp:str = seg["content"]
            o = self.displayRoleText(role, resp)
            out.append(o)
        return "\n".join(out)


    def displayContext(self, max_length=None):
        if not self.display_responses:
            return
        if max_length is None:
            history = self.current_dialog.dialog_history
        else:
            history = self.current_dialog.dialog_history[-max_length:]
        for seg in history:
            role = seg["role"]
            resp:str = seg["content"]
            self.displayRoleText(role, resp)
            if role == "assistant":
                self.parse_shell_commands(resp)

    def displayRoleText(self, role, context):
        if role == "system":
            return self.displaySystemOutput(context)
        elif role == "assistant":
            return self.displayAssistantOutput(context)
        else:
            return self.displayUserInput(context)