
import numpy as np
import os
import datetime
import requests
import csv
from openai import OpenAI
import platform

from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from LLMChatConfig import DEFAULT_MODEL_NAME, DEFAULT_API_URL, DEFAULT_API_KEY, WINDOWS_PLATFORM_NAME, MACOS_PLATFORM_NAME, DEFAULT_DATA_EXTRACT_MODEL, DEFAULT_PRE_TRAINED_TOKENIZER_PATH
from LLMChatConfig import DEFAULT_COMMAND_TEMP, EMBEDDING_MODEL_NAME, DEFAULT_START_DELIMITER, DEFAULT_END_DELIMITER, DEFAULT_DIALOG_TEMP


def user_prompt_segment(text):
    return {"content":text, "role":"user"}

def system_prompt_segment(text):
    return {"content":text, "role":"system"}

def assistant_prompt_segment(text):
    return {"content":text, "role":"assistant"}


current_directory = os.getcwd()  # Initial directory
previous_directory = current_directory  # Track the previous directory for 'cd -'
user_home = os.path.expanduser('~')

def dialog_token_size(dialog_file):
    data = deserialize_from_file(dialog_file)
    total_query = ""

    # TODO: make these headers model-specific
    system_header = "<|start_header_id|>system<|end_header_id|>"
    assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"

    role_header_map = {"assistant":assistant_header, "system":system_header, "user":user_header}
    for seg in data:
        content = seg["content"]
        role = seg["role"]

        total_query+=f"{role_header_map[role]}{content}"
    return get_number_of_llama3_tokens(total_query)

def flatten_dialog_list(dialog_list, use_separate_lines_p = False):
    total_query = ""

    # TODO: make these headers model-specific
    system_header = "system:\n"
    assistant_header = "assistant:\n"
    user_header = "user:\n"

    role_header_map = {"assistant": assistant_header, "system": system_header, "user": user_header}
    for seg in dialog_list:
        content = seg["content"]
        role = seg["role"]
        if use_separate_lines_p:
            total_query += f"{role_header_map[role]}{content}\n"
        else:
            total_query += f"{role_header_map[role]}{content}"
    return total_query


def match_pattern(pattern, user_input, llm_name =DEFAULT_MODEL_NAME, url = DEFAULT_API_URL, api_key = DEFAULT_API_KEY ):
    from LinearLLMDialog import LinearLLMDialog
    import json

    prior = [
        "Your job is to do data extraction from strings and only that.  You will be given a pattern string, P, which will combine ordinary text with pattern capture keys.  The pattern capture keys will be delimited by curly braces and will be a description of the type of text you can expect to find in that section of the string.",
        "You will also be given a raw input string, R, from which you will try to extract substrings in the positions corresponding to the pattern capture keys in P.  If you find substrings that fit that pattern then return a json object whose keys are the pattern capture keys from P that had matching text in R.",
        "For example, if the pattern, P is 'Switch to {directory}' and the input, R, is 'Switch to /usr/local/bin' then you should return: `{\"directory\": \"/usr/local/bin\"}`.  If you can't find any matches in the positions of the capture strings then return only `{}`.  Regardless of how you respond, do not explain your reasoning as it will cause crashes in the system that processes your output.  Also, delimit all responses with backticks, `"]

    examples = []
    examples.append(
        user_prompt_segment(f"Suppose P = '{pattern}' and R = '{user_input}' then your response should be:"))

    model = LinearLLMDialog(prior, prior_user_assist_context=examples, model_name=llm_name, model_url=url, model_api_key=api_key )
    resp = model.chat(f"Suppose P = '{pattern}' and R = '{user_input}' then your response should be:", temp=0.0)
    d = get_delimited_text(resp, "`", "`")[-1][0]
    out = json.loads(d)
    return out

def run_shell_command(command):
    import subprocess
    global current_directory, previous_directory

    # Check if the command is a cd command
    if command.startswith('cd '):
        target_directory = command[3:].strip()

        if target_directory == '-':
            new_directory, current_directory = current_directory, previous_directory
            previous_directory = new_directory
        else:
            # Expand ~ to the home directory
            target_directory = os.path.expanduser(target_directory)
            new_directory = os.path.abspath(os.path.join(current_directory, target_directory))

            # Check if the new directory is valid
            if not os.path.isdir(new_directory):
                return 1, "", f"cd: no such file or directory: {target_directory}\n"

            previous_directory = current_directory
            current_directory = new_directory

        # No need to run this command in subprocess
        return 0, f"Changed directory to {current_directory}\n", ""
    else:
        # Run the command in the current directory'
        if platform.system() == WINDOWS_PLATFORM_NAME:
            result = subprocess.run(["cmd", "/c", command], capture_output=True, text = True, cwd = current_directory)
        elif platform.system() == MACOS_PLATFORM_NAME:
            result = subprocess.run(['zsh', '-c', command], capture_output=True, text=True, cwd=current_directory)
        else:
            result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, cwd=current_directory)
        # Capture the return code, stdout, and stderr
        return_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr

        return return_code, stdout, stderr

def get_short_filename(full_name, ignore_ext = True):
    full_name = os.path.basename(full_name)

    short_filename, ext = os.path.splitext(full_name)
    if ignore_ext:
        return short_filename
    else:
        return short_filename + ext

def get_evaluated_response(exp):
    try:
        res = eval(exp)
        return f"system: {res}"
    except Exception as e:
        try:
            res = exec(exp)
            return f"system: {res}"
        except Exception as e2:
            return f"error: {e} or {e2}"

def get_ordered_date_string(now = None):
    if now is None:
        now = datetime.datetime.now()
    return f"{now.strftime('%Y%m%d')}_{now.strftime('%H')}_{now.strftime('%M')}"


def format_date(epoch_time):
    from datetime import datetime
    dt = datetime.fromtimestamp(epoch_time)
    return f"{dt.strftime('%A, %B')} {dt.day} {dt.year} at {dt.strftime('%I:%M %p')}"


def merge_dialog_roles(dialog_list, role="system", command_delimiter = "\n"):
    out = []
    for seg in dialog_list:
        if len(out) == 0:
            out.append(seg)
        else:
            last = out[-1]
            if role == last["role"]:
                last_content = last["content"]
                new_content = seg["content"]
                last["content"] = f"{last_content}{command_delimiter}{new_content}"
            else:
                out.append(seg)
    return out


def get_plural_form(user_input, model_name = None ):
    if model_name is None:
        model_name = DEFAULT_DATA_EXTRACT_MODEL
    start_delimiter = "<"
    end_delimiter = ">"
    context:list = [{"content":"You are an AI appliance operating as a subsystem within a larger system.  Your output will be processed by a computer, so your responses should be as short and concise as possible", "role":"system"}]
    command:str = f"Return the plural form of the following phrase: \"{user_input}\" using the following start and end delimiters: {start_delimiter} {end_delimiter}"
    res = do_multi_shot_llm_query(context, command, llm_name =  model_name)
    res = get_delimited_text(res, start_delimiter, end_delimiter)
    if res and len(res) > 0:
        return res[0][0]
    else:
        return None



def extract_files_or_folders_from_user_input(user_input, files = True, folders = True, model_name = None):
    out = None
    if model_name is None:
        model_name = DEFAULT_DATA_EXTRACT_MODEL
    start_delimiter = "<"
    end_delimiter = ">"
    no_data_response = "o<>o"
    file_system_os_desc = "Windows or unix"

    target_ref = None
    if files and folders:
        target_ref = "file or folder"
    elif files:
        target_ref = "file"
    elif folders:
        target_ref = "folder"

    if target_ref is None:
        return None

    prior_context = []
    data_extract_system_prompt = "You are an AI appliance operating as a subsystem within a larger system.  Your output will be processed by a computer"
    prior_context.append(system_prompt_segment(data_extract_system_prompt))

    purpose = rf"""You are a simple agent whose purpose is to extract {get_plural_form(target_ref)} from user input.  
    If you can find a reference to a valid {file_system_os_desc} {target_ref}, then return that {target_ref} using the following start and end delimiters, {start_delimiter} {end_delimiter}.  Specify the full path whenever possible.  If you do not find such a reference then return {no_data_response}."""

    prior_context.append(system_prompt_segment(purpose))


    prior_context = merge_dialog_roles(prior_context)
    effective_query = f"For example, if the user says: \"{user_input}\" then return your response is: "
    res = do_multi_shot_llm_query(prior_context, effective_query, llm_name=model_name, temperature = DEFAULT_COMMAND_TEMP)

    if res.find(no_data_response) < 0:
        res = get_delimited_text(res, start_delimiter, end_delimiter)
        if res and len(res) > 0:
            out = res[0][0]
    return out.strip()



def get_number_of_llama3_tokens(input):
    if platform.platform() == "Windows":
        tokenizer = PreTrainedTokenizer.from_pretrained(DEFAULT_PRE_TRAINED_TOKENIZER_PATH)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained( DEFAULT_PRE_TRAINED_TOKENIZER_PATH)

    def count_tokens(text):
        # Tokenize the input text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Return the number of tokens
        return len(tokens)

    return count_tokens(input)

def read_prompt_file(prompt_file_name):
    with open(prompt_file_name, 'r') as file:
        return file.read()



def write_string_to_file(string, filename):
    with open(filename, 'w') as f:
        f.write(string)



def is_llm_server_available(url = DEFAULT_API_URL, api_key = DEFAULT_API_KEY):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        return False


def get_available_models_from_llm_server(url = DEFAULT_API_URL,  api_key = DEFAULT_API_KEY):
    model_url = f"{url}/models"
    api_key = api_key

    # Set up the headers with the Authorization token
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(model_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return [model["id"] for model in data["data"]]
    else:
        return None



def get_closest_semantic_match(query, options, model=EMBEDDING_MODEL_NAME,url = DEFAULT_API_URL, api_key = DEFAULT_API_KEY):
    choice_embedding = get_embedding(query, model, url, api_key)
    distances = np.array([np.linalg.norm(get_embedding(option) - choice_embedding) for option in options])
    return np.argmin(distances)

def get_embedding(text, model=EMBEDDING_MODEL_NAME,url = DEFAULT_API_URL, api_key = DEFAULT_API_KEY):
    client = OpenAI(base_url=url, api_key=api_key)
    text = text.replace("\n", " ")
    return np.array(client.embeddings.create(input = [text], model=model).data[0].embedding)



def apply_custom_delimiter(text, delimiter_start = DEFAULT_START_DELIMITER, delimiter_end = DEFAULT_END_DELIMITER):
    return f"{delimiter_start}{text}{delimiter_end}"

def get_delimited_text(text, delimiter_start = DEFAULT_START_DELIMITER, delimiter_end = DEFAULT_END_DELIMITER):
    strings = []
    slength =len(delimiter_start)
    elength = len(delimiter_end)
    start_delimiter_pos = text.find(delimiter_start)
    end_delimiter_pos = text.find(delimiter_end, start_delimiter_pos + slength)
    while start_delimiter_pos >= 0 and end_delimiter_pos > start_delimiter_pos:
        substring = text[(start_delimiter_pos+slength):end_delimiter_pos]
        strings.append((substring, start_delimiter_pos, end_delimiter_pos+elength))
        text = text[(end_delimiter_pos+elength):]
        start_delimiter_pos = text.find(delimiter_start)
        end_delimiter_pos = text.find(delimiter_end, start_delimiter_pos + slength)
    return strings

def get_system_prompt_for_string_literals(delimiter_start = DEFAULT_START_DELIMITER, delimiter_end = DEFAULT_END_DELIMITER):
    system_prompt = fr"""Raw string text must be processed according to a set of rules that will be described to you. 
    Raw strings are delimited by a string start token: {delimiter_start} and an ending token: {delimiter_end}.  That way, the raw 
    string text like "hello there!" will be represented as {delimiter_start}hello there!{delimiter_end}.  The idea is that I chose
     strings that are unlikely to naturally occur in any input as the string delimiter tokens to avoid confusion.  Having said that, 
      any time you need to describe the literal contents of a file or string, you must use that convention in your responses.
    """
    return {"content":system_prompt, "role":"system"}




def do_multi_shot_llm_query(prior_dialog_history, query, llm_name =DEFAULT_MODEL_NAME, verbose = False, url = DEFAULT_API_URL, api_key = DEFAULT_API_KEY, temperature = DEFAULT_COMMAND_TEMP):
    if prior_dialog_history is None:
        prior_dialog_history = []
        prior_dialog_history.append(get_system_prompt_for_string_literals())
    prior_dialog_history.append({"role": "user", "content": query})
    client = OpenAI(base_url=url, api_key=api_key)

    model_hugging_face_name = llm_name
    completion = client.chat.completions.create(
        model=model_hugging_face_name,
        messages=prior_dialog_history,
        temperature=temperature,
    )

    response_message = completion.choices[0].message

    prior_dialog_history.append({"role": "assistant", "content": response_message.content})
    return response_message.content

def process_llm_history(prior_dialog_history, llm_name =DEFAULT_MODEL_NAME, verbose = False, url = DEFAULT_API_URL, api_key = DEFAULT_API_KEY, temperature = DEFAULT_COMMAND_TEMP):
    client = OpenAI(base_url=url, api_key=api_key)

    model_hugging_face_name = llm_name
    completion = client.chat.completions.create(
        model=model_hugging_face_name,
        messages=prior_dialog_history,
        temperature=temperature,
    )

    response_message = completion.choices[0].message

    prior_dialog_history.append({"role": "assistant", "content": response_message.content})
    return response_message.content

def do_one_shot_llm_query(query, delimiter_start = DEFAULT_START_DELIMITER, delimiter_end = DEFAULT_END_DELIMITER, llm_name = DEFAULT_MODEL_NAME, verbose = False, url = DEFAULT_API_URL, api_key = DEFAULT_API_KEY, temperature = DEFAULT_DIALOG_TEMP):
    system_prompt = get_system_prompt_for_string_literals(delimiter_start, delimiter_end)
    if verbose:
        print(f"Base system prompt:")
        print(system_prompt+"\n\n")

    client = OpenAI(base_url=url, api_key=api_key)

    model_hugging_face_name = llm_name
    completion = client.chat.completions.create(
        model=model_hugging_face_name,
        messages=[
            {"role": "system", "content": system_prompt["content"]},
            {"role": "user", "content": query}
        ],
        temperature=temperature,
    )

    response_message = completion.choices[0].message
    return response_message.content


def serialize_to_file(data, filename):
    """
    Serialize the given data (a list of dictionaries) to a JSON file.

    Args:
        data (list): A list of dictionaries.
        filename (str): The name of the file to write the serialized data to.
    """
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)

def deserialize_from_file(filename):
    """
    Deserialize a JSON file into a list of dictionaries.

    Args:
        filename (str): The name of the file to read from.

    Returns:
        list: A list of dictionaries.
    """
    import json
    if not os.path.exists(filename):
        return []

    with open(filename, 'r') as f:
        data = json.load(f)
        return data

def get_numerated_list_prompt(options:list[str],include_header_footer = None, quote_items = None) ->str:

    if quote_items is None:
        quote_items = False
    if include_header_footer is None:
        include_header_footer = True
    if include_header_footer:
        out = "---------------\n"
    else:
        out = ""
    for i, option_desc in enumerate(options):
        if quote_items:
            options = f"({i+1}) \"{option_desc}\"\n"
        else:
            options = f"({i+1}) {option_desc}\n"
        out += options
    if include_header_footer:
        out += "---------------\n"
    return out



def parse_dict_from_csv(csv_file:str) -> dict:
    data = csv.reader(open(csv_file))
    command_specs = list(data)[1:]
    out = {}

    for command_type, com_variant in command_specs:
        if command_type in out:
            out[command_type].append(com_variant)
        else:
            out[command_type] = [com_variant.strip()]
    return out


def parse_command_map(command_map_file:str) -> dict:
    return parse_dict_from_csv(command_map_file)

def parse_test_input_map(test_input_csv_file:str) -> dict:
    return parse_dict_from_csv(test_input_csv_file)


def get_embedding_centroid(vect_list:list[np.ndarray]):
    return np.array(vect_list).mean(axis = 0)
def get_command_embedding_map(command_map:dict[str, list[str]], model, url, api_key) -> dict:
    out = {}
    for command_type_key, command_variants in command_map.items():
        embeddings = [get_embedding(variant, model, url, api_key) for variant in command_variants]
        variant_centroid = get_embedding_centroid(embeddings)
        out[command_type_key] = variant_centroid

    return out

def get_input_variant_embedding_map(test_eg_map:dict[str, list[str]], model, url, api_key) -> dict[str, list[np.ndarray]]:
    out = {}

    for (command_type, input_variants) in test_eg_map.items():
        out[command_type] = [get_embedding(variant, model, url, api_key) for variant in input_variants]
    return out


def split_role_message(input):
    import re
    pattern = r"^\s*((([\w,\-\d]+)\:+)\**)(.*)"
    match = re.search(pattern, input)
    role = None
    message = input
    if match:
        role = match.group(1)
        message = match.group(4).strip()
        if not message:
            message = None

    return (role, message)


def parse_roles_from_dialog_string(dialog_str:str, use_llm_input_structure_p = False):
    dialog = []
    previous_role = None

    for line in dialog_str.split("\n"):
        if (len(line.strip())>0):
            role, message = split_role_message(line.strip())
            if role:
                base_role = role.strip("*")
                is_preferred = len(role) > len(base_role)
                if previous_role is None:
                    dialog.append((base_role, [(is_preferred, message)]))
                elif previous_role == base_role:
                    all_role_messages = dialog[-1][1]
                    all_role_messages.append((is_preferred, message))
                else:
                    if message is None:
                        message = ""
                    dialog.append((base_role, [(is_preferred, message)]))
                previous_role = base_role
            else:
                if len(dialog)>0:
                    all_role_messages = dialog[-1][1]
                    message_pref, message_being_updated = all_role_messages[-1]
                    all_role_messages[-1] = (message_pref, message_being_updated + "\n" + message)
    if use_llm_input_structure_p:
        def to_llm_structure(item_spec):
            role:str = item_spec[0][0:-1]
            message:str = item_spec[1][0][1]
            return {"content":message, "role":role.lower()}

        return [to_llm_structure(item) for item in dialog]
    else:
        return dialog
def parse_roles_from_dialog_pattern_file(dialog_pattern_context_file, use_llm_input_structure_p = False):
    # dialog consists of a list of tuples, (base_role:str, message:list[(bool, str)])
    dialog = []
    previous_role = None
    with open(dialog_pattern_context_file, "r") as w:
        for line in w:
            role, message = split_role_message(line.strip())
            if role:
                base_role = role.strip("*")
                is_preferred = len(role) > len(base_role)
                if previous_role is None:
                    dialog.append((base_role, [(is_preferred, message)]))
                elif previous_role == base_role:
                    all_role_messages = dialog[-1][1]
                    all_role_messages.append((is_preferred, message))
                else:
                    dialog.append((base_role, [(is_preferred, message)]))
                previous_role = base_role
            else:
                all_role_messages = dialog[-1][1]
                message_pref, message_being_updated = all_role_messages[-1]
                all_role_messages[-1] = (message_pref, message_being_updated + "\n" + message)
    if use_llm_input_structure_p:
        def to_llm_structure(item_spec):
            role:str = item_spec[0][0:-1]
            message: str = item_spec[1][0][1]
            return {"content": message, "role": role.lower()}

        return [to_llm_structure(item) for item in dialog]
    else:
        return dialog