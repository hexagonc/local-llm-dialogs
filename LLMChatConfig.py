import os
import platform
import random
from typing import Optional
import uuid
import random
from typing import Optional
import json

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

DEFAULT_LLM_NAME_CONFIG_KEY = "default-model-name"
DEFAULT_LLM_API_KEY_CONFIG_KEY = "default-model-api-key"
DEFAULT_LLM_URL_CONFIG_KEY = "default-model-api-url"

DEFAULT_EMBEDDING_MODEL_NAME_CONFIG_KEY = "default-embedding-model-name"
DEFAULT_EMBEDDING_MODEL_URL_CONFIG_KEY = "default-embedding-model-url"
DEFAULT_EMBEDDING_MODEL_API_KEY_CONFIG_KEY = "default-embedding-model-api-key"

DEFAULT_PRE_TRAINED_TOKENIZER_PATH_CONFIG_KEY = "pretrained-tokenizer-path"

CONFIG_FILE_NAME = "config.json"

CONFIG_MAP = json.loads(read_text(CONFIG_FILE_NAME))

def get_config_str(key, default = None) -> Optional[str]:
    if key in CONFIG_MAP:
        return CONFIG_MAP[key]
    else:
        return default

LM_STUDIO_API_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LLAMA_LLM_NAME = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

DEFAULT_PRE_TRAINED_TOKENIZER_PATH = get_config_str(DEFAULT_PRE_TRAINED_TOKENIZER_PATH_CONFIG_KEY, "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")

EMBEDDING_MODEL_NAME = get_config_str(DEFAULT_EMBEDDING_MODEL_NAME_CONFIG_KEY, "nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf")

DEFAULT_MODEL_NAME = get_config_str(DEFAULT_LLM_NAME_CONFIG_KEY, LLAMA_LLM_NAME)
DEFAULT_API_URL = get_config_str(DEFAULT_LLM_URL_CONFIG_KEY, LM_STUDIO_API_URL)
DEFAULT_API_KEY = get_config_str(DEFAULT_LLM_API_KEY_CONFIG_KEY, LM_STUDIO_API_KEY)

DEFAULT_DATA_EXTRACT_MODEL =  DEFAULT_MODEL_NAME

DEFAULT_DIALOG_TEMP = 0.7

DEFAULT_COMMAND_TEMP = 0.0

def set_default_model(model):
    global DEFAULT_MODEL_NAME
    DEFAULT_MODEL_NAME = model

def set_data_extract_model(model):
    global DEFAULT_DATA_EXTRACT_MODEL
    DEFAULT_DATA_EXTRACT_MODEL = model
