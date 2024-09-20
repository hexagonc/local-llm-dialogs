import json
import numpy as np
import os
import datetime
import requests
import uuid
from openai import OpenAI
import random
from typing import Optional
import platform

from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

PATH_SEP = os.sep
WINDOWS_PLATFORM_NAME = "Windows"
MACOS_PLATFORM_NAME = "Darwin"
LINUX_PLATFORM_NAME = "Linux"

PLATFORM_NAME = platform.system()

PLATFORM_DESC = {WINDOWS_PLATFORM_NAME:WINDOWS_PLATFORM_NAME, MACOS_PLATFORM_NAME:"MacOS", LINUX_PLATFORM_NAME:LINUX_PLATFORM_NAME}[PLATFORM_NAME]
LINUX_DEFAULT_SHELL_NAME = "bash"
MACOS_DEFAULT_SHELL_NAME = "zsh"
WINDOWS_DEFAULT_SHELL_NAME = "DOS"

DEFAULT_PLATFORM_SHELL_NAME={WINDOWS_PLATFORM_NAME:WINDOWS_DEFAULT_SHELL_NAME, MACOS_PLATFORM_NAME:MACOS_DEFAULT_SHELL_NAME, LINUX_PLATFORM_NAME:LINUX_DEFAULT_SHELL_NAME}[PLATFORM_NAME]

default_text_file_ext_list = ['.txt', '.md', '.rst', '.html', '.htm', '.xml', '.json', '.csv', '.tsv', '.yaml', '.yml',
                              '.log', '.ini', '.cfg', '.conf', '.properties', '.java', '.js', '.ts', '.py', '.sh',
                              '.bat', '.cmd', '.ps1', '.psm1', '.psd1', '.ps1xml', '.pssc', '.pssc', '.pss', '.gradle']
default_text_file_ext_set = set(default_text_file_ext_list)


delimiter_seed = 90

def seeded_uuid(seed):
    random.seed(seed)
    return uuid.UUID(int=random.getrandbits(128))

def read_text(f):
    with open(f, 'r') as file:
        return file.read()

reproducible_uuid = seeded_uuid(delimiter_seed)

delimiter_id = str(reproducible_uuid)[:5]
DEFAULT_START_DELIMITER = f"[{delimiter_id}"
DEFAULT_END_DELIMITER = f"{delimiter_id}]"


# Config file key names
OPENAI_API_KEY_CONFIG_KEY = "openai-api-key"
OPENAI_API_URL_CONFIG_KEY = "openai-api-url"

DEFAULT_LLM_NAME_CONFIG_KEY = "default-model-name"
DEFAULT_LLM_API_KEY_CONFIG_KEY = "default-model-api-key"
DEFAULT_LLM_URL_CONFIG_KEY = "default-model-api-url"

DEFAULT_EMBEDDING_MODEL_NAME_CONFIG_KEY = "default-embedding-model-name"
DEFAULT_EMBEDDING_MODEL_URL_CONFIG_KEY = "default-embedding-model-url"
DEFAULT_EMBEDDING_MODEL_API_KEY_CONFIG_KEY = "default-embedding-model-api-key"

DEFAULT_PRE_TRAINED_TOKENIZER_PATH_CONFIG_KEY = "pretrained-tokenizer-path"

CONFIG_FILE_NAME = "config.json"

CONFIG_MAP = json.loads(read_text(CONFIG_FILE_NAME))

CURRENT_DIRECTORY = os.getcwd()

def get_config_str(key, default = None) -> Optional[str]:
    if key in CONFIG_MAP:
        return CONFIG_MAP[key]
    else:
        return default

DEFAULT_MODEL_COMMAND_CONFIG = {}

if "model-command-config" in CONFIG_MAP:
    DEFAULT_MODEL_COMMAND_CONFIG = CONFIG_MAP["model-command-config"]

LM_STUDIO_API_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LLAMA_LLM_NAME = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

DEFAULT_PRE_TRAINED_TOKENIZER_PATH = get_config_str(DEFAULT_PRE_TRAINED_TOKENIZER_PATH_CONFIG_KEY, "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")

EMBEDDING_MODEL_NAME = get_config_str(DEFAULT_EMBEDDING_MODEL_NAME_CONFIG_KEY, "nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf")

DEFAULT_MODEL_NAME = get_config_str(DEFAULT_LLM_NAME_CONFIG_KEY, LLAMA_LLM_NAME)
DEFAULT_API_URL = get_config_str(DEFAULT_LLM_URL_CONFIG_KEY, LM_STUDIO_API_URL)
DEFAULT_API_KEY = get_config_str(DEFAULT_LLM_API_KEY_CONFIG_KEY, LM_STUDIO_API_KEY)

DEFAULT_DATA_EXTRACT_MODEL =  DEFAULT_MODEL_NAME

OPENAI_API_URL = get_config_str(OPENAI_API_URL_CONFIG_KEY, "https://api.openai.com/v1")
OPENAI_API_KEY = get_config_str(OPENAI_API_KEY_CONFIG_KEY, "")
OPENAI_CHAT_GPT_TURBO = "gpt-3.5-turbo-instruct"
OPENAI_GPT4 = "gpt-4-turbo"
OPENAI_GPT4o = "gpt-4o"

DEFAULT_DIALOG_TEMP = 0.7

DEFAULT_COMMAND_TEMP = 0.0

def set_default_model(model):
    global DEFAULT_MODEL_NAME
    DEFAULT_MODEL_NAME = model

def set_data_extract_model(model):
    global DEFAULT_DATA_EXTRACT_MODEL
    DEFAULT_DATA_EXTRACT_MODEL = model

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

def flatten_dialog_list(dialog_list):
    total_query = ""

    # TODO: make these headers model-specific
    system_header = "system:\n"
    assistant_header = "assistant:\n"
    user_header = "user:\n"

    role_header_map = {"assistant": assistant_header, "system": system_header, "user": user_header}
    for seg in dialog_list:
        content = seg["content"]
        role = seg["role"]
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


def serialize_to_file(data, filename, pretty = False):
    """
    Serialize the given data (a list of dictionaries) to a JSON file.

    Args:
        data (list): A list of dictionaries.
        filename (str): The name of the file to write the serialized data to.
    """
    import json
    with open(filename, 'w') as f:
        if pretty:
            json.dump(data, f, indent=4, sort_keys=True)
        else:
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