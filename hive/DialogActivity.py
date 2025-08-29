import datetime
import time
from typing import Optional

from LLMTools import read_prompt_file, write_string_to_file, parse_roles_from_dialog_pattern_file, \
    parse_roles_from_dialog_string, do_multi_shot_llm_query
import json
import os
from plisp.Environment import Environment
from plisp.LispTools import add_basic_functions, add_arithmetic_functions

from plisp.FileTools import add_filesystem_functions
from .Hive import DIALOG_SEGMENT_BASE, DIALOG_SEGMENT_EXTERNAL_INTERFACE, DIALOG_SEGMENT_LEARNED_RULES, \
    DIALOG_SEGMENT_PLISP, DIALOG_SEGMENT_EXPERIENCE, DIALOG_SEGMENT_HIVE_COMMUNICATION, DIALOG_SEGMENT_TASK
from .HiveUtils import build_prompt_from_template, parse_epoch_from_standard_datetime_str, \
    get_standard_datetime_str_from_epoch, get_file_datetime
from .MetaRoleHandler import MetaRoleHandler

DIALOG_CONFIG_HEAD_KEY = ""
DIALOG_CONFIG_TOPIC_KEY = "topic"
META_ROLE_NAME_KEY = "meta-role-prefix"
DIALOG_TOPIC_NAME_SHORT_KEY = "simple-topic-name"
DIALOG_BASE_PROMPT_KEY = "dialog-base-prompt"
DIALOG_CONFIG_PURPOSE_KEY = "dialog-purpose"
DIALOG_CONFIG_HEADER_KEY = "dialog-rules"
DIALOG_CONFIG_LISP_ENV_KEY = "dialog-lisp-env"
DIALOG_HEADS_KEY = "dialog-experience"

from LLMChatConfig import CONFIG_MAP


class DialogActivity(MetaRoleHandler):
    tag = "o_DialogActivity_o"
    verbose = True

    def __init__(self, hive, dialog_topic, dialog_short_name, default_role_prefix, dialog_config_file = None, base_path = "dialogs"):
        super().__init__(default_role_prefix, dialog_topic)
        self.dialog_model_logical_name = "dialog-tasks"
        self.model_name, self.url, self.api_key = CONFIG_MAP["model-config"][self.dialog_model_logical_name]
        self.base_path = base_path
        self.hive = hive
        self.segment_map = {}

        if dialog_config_file is None:
            dialog_config_file = f"{base_path}/{dialog_short_name}.json"

        self.dialog_config_file = dialog_config_file

        if os.path.exists(dialog_config_file):
            self.dialog_config = json.loads(read_prompt_file(dialog_config_file))
        else:
            self.dialog_config = {DIALOG_CONFIG_TOPIC_KEY: dialog_topic,
	                              META_ROLE_NAME_KEY: default_role_prefix,
	                              DIALOG_TOPIC_NAME_SHORT_KEY: dialog_short_name,
	                              DIALOG_BASE_PROMPT_KEY: f"{dialog_short_name}_common.txt",
	                              DIALOG_CONFIG_PURPOSE_KEY: f"{dialog_short_name}_purpose.txt",
	                              DIALOG_CONFIG_HEADER_KEY: f"{dialog_short_name}_learned_rules.txt",
	                              DIALOG_CONFIG_LISP_ENV_KEY: f"{dialog_short_name}_env.lisp",
                                  DIALOG_HEADS_KEY: []}
        self.topic = self.dialog_config[DIALOG_CONFIG_TOPIC_KEY]
        self.meta_role_name = self.dialog_config[META_ROLE_NAME_KEY]
        self.short_topic_name = self.dialog_config[DIALOG_TOPIC_NAME_SHORT_KEY]

        filename_short = self.dialog_config[DIALOG_BASE_PROMPT_KEY]
        filename = f"{base_path}/{filename_short}"
        if os.path.exists(filename):
            self.base_prompt = read_prompt_file(filename)
        else:
            self.base_prompt = hive.get_prompt_segment(DIALOG_SEGMENT_BASE)

        self.segment_map[DIALOG_SEGMENT_BASE] = self.base_prompt

        filename_short = self.dialog_config[DIALOG_CONFIG_PURPOSE_KEY]
        filename = f"{base_path}/{filename_short}"
        if os.path.exists(filename):
            self.dialog_purpose = read_prompt_file(filename)
            self.on_connect_message = parse_roles_from_dialog_string(self.dialog_purpose, True)[0]
            self.segment_map[DIALOG_SEGMENT_EXTERNAL_INTERFACE] = self.dialog_purpose

        filename_short = self.dialog_config[DIALOG_CONFIG_HEADER_KEY]
        filename = f"{base_path}/{filename_short}"
        if os.path.exists(filename):
            self.dialog_lessons = read_prompt_file(filename)
            self.segment_map[DIALOG_SEGMENT_LEARNED_RULES] = self.dialog_lessons

        env = Environment(hive.get_environment())
        filename_short = self.dialog_config[DIALOG_CONFIG_LISP_ENV_KEY]
        filename = f"{base_path}/{filename_short}"
        if os.path.exists(filename):
            data = read_prompt_file(filename)

            if len(data.strip())>0:
                env.fromSerialized(data.strip())
        else:
            add_basic_functions(env)
            add_arithmetic_functions(env)
            add_filesystem_functions(env)
        from .HiveUtils import to_svalue
        env.map_value("meta-role", to_svalue(self.meta_role_name))
        env.map_value("topic", to_svalue(self.topic))

        self.dialog_experience_segments = []
        self.experience_segment = {DIALOG_SEGMENT_TASK}
        for dialog_start_datetime_str, filename in self.dialog_config[DIALOG_HEADS_KEY]:
            prompt = read_prompt_file(f"{base_path}/{filename_short}")
            datetime_epoch = parse_epoch_from_standard_datetime_str(dialog_start_datetime_str)
            self.dialog_experience_segments.append((datetime_epoch, prompt))
            self.segment_map[dialog_start_datetime_str] = prompt
            self.experience_segment.add(dialog_start_datetime_str)

        self.dialog_env = env

        self.allow_updating_non_experience_segments = False

        self.model_config = CONFIG_MAP["model-config"]

    def get_environment(self):
        return self.dialog_env

    def start_updating_non_experience_segments(self):
        self.allow_updating_non_experience_segments = True

    def stop_updating_non_experience_segments(self):
        self.allow_updating_non_experience_segments = False

    def get_all_experience_prompt(self):
        total_prompt = ""
        for dt, exp_prompt in self.dialog_experience_segments:
            if len(total_prompt) > 0:
                total_prompt += "\n"
            total_prompt += exp_prompt
        return total_prompt

    def chat_with_segments(self, message:str, active_segments:[str], mutate_non_experience_segments_p = None):
        if mutate_non_experience_segments_p is None:
            mutate_non_experience_segments_p = self.allow_updating_non_experience_segments

        prompt_replacements = {DIALOG_SEGMENT_EXPERIENCE:self.get_all_experience_prompt()}
        prompt_template = ""
        last_updated_segment = None
        for segment_name in active_segments:
            if segment_name in self.segment_map:
                prompt_template += self.segment_map[segment_name]
                last_updated_segment = segment_name
            elif segment_name in prompt_replacements:
                prompt_template += prompt_replacements[segment_name]
                last_updated_segment = segment_name
            else:
                hive_prompt = self.hive.get_prompt_segment(segment_name)
                if hive_prompt:
                    prompt_template += hive_prompt
                    last_updated_segment = segment_name
        effective_prompt = build_prompt_from_template(self.dialog_env, prompt_template)
        dialog = parse_roles_from_dialog_string(effective_prompt, True)
        response = do_multi_shot_llm_query(dialog, message, llm_name=self.model_name, url=self.url,
                                           api_key=self.api_key)
        if last_updated_segment:
            if last_updated_segment in self.experience_segment:
                self.segment_map[last_updated_segment] += f"\nuser: " + message
                self.segment_map[last_updated_segment] += f"\nassistant: " + response
            elif mutate_non_experience_segments_p:
                self.segment_map[last_updated_segment] += f"\nuser: " + message
                self.segment_map[last_updated_segment] += f"\nassistant: " + response
            else:
                self.segment_map[DIALOG_SEGMENT_TASK] = f"\nuser: {message}\nassistant: {response}"
        return response

    def user_chat(self, text:str, add_new_task = False):
        active_chat_segments = [DIALOG_SEGMENT_BASE, DIALOG_SEGMENT_HIVE_COMMUNICATION, DIALOG_SEGMENT_PLISP]
        append_segment= []

        if len(self.dialog_experience_segments) > 0:
            # Latest segment is the last
            ts, prompt = self.dialog_experience_segments[-1]
            resolved_prompt = build_prompt_from_template(self.dialog_env, prompt)
            append_segment = parse_roles_from_dialog_string(resolved_prompt, True)
        else:
            segment_create_datetime_ms = time.time()*1000
            segment_name = get_standard_datetime_str_from_epoch(segment_create_datetime_ms)

            self.dialog_experience_segments.append((segment_create_datetime_ms, append_segment))




    def save_current_task(self, clear_task_p = True):
        if DIALOG_SEGMENT_TASK in self.segment_map:
            current_date = get_standard_datetime_str_from_epoch()
            task_history = self.segment_map[DIALOG_SEGMENT_TASK]
            self.segment_map[current_date] = task_history
            if clear_task_p:
                self.segment_map.remove(DIALOG_SEGMENT_TASK)
            time_epoch = parse_epoch_from_standard_datetime_str(current_date)
            filename_time = get_file_datetime(time_epoch)
            new_filename = f"{self.short_topic_name}_history_from_{filename_time}.txt"
            self.dialog_config[DIALOG_HEADS_KEY].append((current_date, new_filename))
            self.dialog_experience_segments.append((current_date, task_history))
            self.experience_segment.add(current_date)


    def chat(self, client_meta_role:str, message:str, client_env:Optional[Environment] = None) -> str:
        if len(message.strip()) == 0 and DIALOG_SEGMENT_EXTERNAL_INTERFACE in self.segment_map:
            return f"{client_meta_role}:" + self.segment_map[DIALOG_SEGMENT_EXTERNAL_INTERFACE]
        if client_meta_role == self.hive.user_role:
            client_meta_role = ""
        else:
            client_meta_role = client_meta_role+":"
        default_message_segments = [DIALOG_SEGMENT_BASE, DIALOG_SEGMENT_HIVE_COMMUNICATION, DIALOG_SEGMENT_PLISP, DIALOG_SEGMENT_LEARNED_RULES, DIALOG_SEGMENT_EXTERNAL_INTERFACE, DIALOG_SEGMENT_TASK]
        return self.chat_with_segments(client_meta_role + message, default_message_segments)

    def saveState(self):
        write_string_to_file(json.dumps(self.dialog_config), self.dialog_config_file)

        base_path = self.base_path
        filename_short = self.dialog_config["dialog-base-prompt"]
        filename = f"{base_path}/{filename_short}"
        write_string_to_file(self.base_prompt, filename)

        filename_short = self.dialog_config["dialog-purpose"]
        filename = f"{base_path}/{filename_short}"
        write_string_to_file(self.dialog_purpose, filename)


        filename_short = self.dialog_config["dialog-base-header"]
        filename = f"{base_path}/{filename_short}"
        write_string_to_file(self.dialog_base_header, filename)

        filename_short = self.dialog_config["dialog-lisp-env"]
        filename = f"{base_path}/{filename_short}"
        write_string_to_file(self.dialog_env.serialize(), filename)

