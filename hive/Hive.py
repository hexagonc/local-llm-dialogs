from typing import Optional
from .NoeticLogger import NoeticLogger
from plisp.Environment import Environment
from plisp.LispTools import add_basic_functions, add_arithmetic_functions
from plisp.FileTools import add_filesystem_functions

from .MetaRoleHandler import MetaRoleHandler
from LLMTools import read_prompt_file, parse_roles_from_dialog_string, do_multi_shot_llm_query, user_prompt_segment, assistant_prompt_segment, process_llm_history
from .HiveUtils import build_prompt_from_template, to_svalue
from LLMChatConfig import CONFIG_MAP

from LLMTools import split_role_message
from plisp.Value import  Value


DIALOG_SEGMENT_BASE = "base"
DIALOG_SEGMENT_PLISP = "plisp"
DIALOG_SEGMENT_EXPERIENCE = "all-exp"
DIALOG_SEGMENT_LEARNED_RULES = "learned-rules"
DIALOG_SEGMENT_EXTERNAL_INTERFACE = "purpose"
DIALOG_SEGMENT_HIVE_COMMUNICATION = "general-communication-protocol"
DIALOG_SEGMENT_TASK = "current-task"


DIALOG_SYSTEM_META_ROLE = "system-manager"
DIALOG_SYSTEM_ROLE_TOPIC = "executes commands against the overall LLM dialog system environment as well as returns system error messages, warnings and instructions"
DIALOG_SYSTEM_TOPIC_SHORT = "system_control_manager"


class Hive(MetaRoleHandler):
    hive_logging_tag = "o_HIVE_o"
    verbose = True
    def __init__(self, base_prompt_file = None, env = None, logger:NoeticLogger = None):
        super().__init__("hive", "top level conversation about global configuration and meta state")

        self.max_nested_llm_auto_call = 10
        if logger is None:
            self.logger = NoeticLogger("hive_logs")
        else:
            self.logger = logger

        self.hive_model_logical_name = "hive-internal"
        self.model_name, self.url, self.api_key = CONFIG_MAP["model-config"][self.hive_model_logical_name]

        if base_prompt_file is None:
            self.base_prompt = read_prompt_file("/Users/evolvedgpt/development/projects/local-llm-dialogs/dialogs/hive_base_prompt.txt")
        else:
            self.base_prompt = read_prompt_file(base_prompt_file)

        self.user_role = "user"
        if env:
           self.hive_env = env
        else:
            self.hive_env = Environment()
            add_basic_functions(self.hive_env)
            add_filesystem_functions(self.hive_env)
            add_arithmetic_functions(self.hive_env)

        self.hive_env.map_value("meta-role", to_svalue(self.meta_role))
        self.hive_env.map_value("topic", to_svalue(self.topic))
        self.dialog_experience = []
        self.state_permission_required_for_next_command = False
        self.state_handling_permission_requested_state = False

        self.role_handlers = {}
        self.prior_role = self.meta_role
        self.hive_env.map_value("prior-role", to_svalue(self.prior_role))

        self.add_hive_local_functions(self.hive_env)
        self.base_hive_topic_creation_rules_fname = "/Users/evolvedgpt/development/projects/local-llm-dialogs/dialogs/base_hive_topic_rules.txt"
        self.base_hive_topic_creation_rules = read_prompt_file(self.base_hive_topic_creation_rules_fname)
        self.base_hive_topic_search_rules_fnamne = "/Users/evolvedgpt/development/projects/local-llm-dialogs/dialogs/base_hive_topic_search_rules.txt"
        self.base_hive_topic_search_rules = read_prompt_file(self.base_hive_topic_search_rules_fnamne)

        self.dialog_segments ={}
        self.dialog_segments[DIALOG_SEGMENT_PLISP] = read_prompt_file("/Users/evolvedgpt/development/projects/local-llm-dialogs/dialogs/plisp_service_desc.txt")
        self.dialog_segments[DIALOG_SEGMENT_HIVE_COMMUNICATION] = read_prompt_file("/Users/evolvedgpt/development/projects/local-llm-dialogs/dialogs/hive_communication_protocol.txt")
        # this is different from base_prompt_file in general since this is used by DialogActivity and not necessarily
        # thie main hive LLM
        self.dialog_segments[DIALOG_SEGMENT_BASE] = read_prompt_file("/Users/evolvedgpt/development/projects/local-llm-dialogs/dialogs/dialog_activity_base_prompt.txt")

        # Add system meta-role handlers
        self.add_meta_role_handler(self, self.topic, self.meta_role, self.meta_role)
        from .PLispMetaRoleHandler import PLispMetaRoleHandler
        plisp = PLispMetaRoleHandler(self)
        plisp.add_self_to_hive(self)
        from .SystemControlMetaHandler import SystemControlMetaHandler
        system = SystemControlMetaHandler(self)
        system.add_self_to_hive(self)
        self.system_error_handler = system


    def add_meta_role_handler(self, handler:MetaRoleHandler, topic:str, topic_name_short:str, meta_role_key:str ):
        self.role_handlers[meta_role_key] = (handler, topic, topic_name_short, meta_role_key)

    def get_environment(self):
        return self.hive_env


    def get_prompt_segment(self, name:str):
        if name in self.dialog_segments:
            return self.dialog_segments[name]
        else:
            return None

    def add_hive_local_functions(self, env:Environment):
        from plisp.SimpleFunctionTemplate import SimpleFunctionTemplate
        def get_existing_topics(template, evaluated_args):
            out = "| meta-role key | topic | topic name short |\n| --- | --- | --- |"
            for role, (handler, topic, topic_name_short, meta_role) in self.role_handlers.items():
                out += f"\n|{role}|{topic}|{topic_name_short}|"

            return to_svalue(out)

        env.map_function_template(SimpleFunctionTemplate("print-all-meta-roles-as-table", get_existing_topics))

        fname = "switch-to-topic"
        def switch_to_topic(template, evaluated_args:[Value]):

            if len(evaluated_args) > 0 and evaluated_args[0].is_string():
                topic:str = evaluated_args[0].string()
                no_topic_response = f"Could not find a definitive match for: '{topic}'"
                existing = self.find_existing_topic(topic, no_topic_response, 2)

                if len(existing) == 0:
                    return no_topic_response
                else:
                    self.prior_role = existing[0]
                    return to_svalue(existing[0])
            else:
                raise Exception(f"First argument to {fname} must be a string topic description.")

        env.map_function_template(SimpleFunctionTemplate(fname, switch_to_topic))


        fname = "create-topic"
        def create_topic(template, evaluated_args:[Value]):

            if len(evaluated_args) > 0 and evaluated_args[0].is_string():
                topic:str = evaluated_args[0].string()
                no_topic_response = f"Unable to create new topic described by {topic}.  Can you be more specific?"
                switch_to_new_activity = len(evaluated_args)>1 and not evaluated_args[1].is_null()
                new_activty, new_role = self.add_dialog_activity(topic)
                if new_role:
                    if switch_to_new_activity:
                        self.prior_role = new_role
                        return to_svalue(f"Created new topic {topic} with role {new_role} and switching to it")
                    else:
                        return to_svalue(f"Created new topic {topic}")
                else:
                    return to_svalue(no_topic_response)
            else:
                raise Exception(f"First argument to {fname} must be a string topic description.")

        env.map_function_template(SimpleFunctionTemplate(fname, create_topic))


        fname = "find-topics-from_description"
        def find_topics(template, evaluated_args:[Value]):

            if len(evaluated_args) > 0 and evaluated_args[0].is_string():
                description:str = evaluated_args[0].string()
                no_topic_response = f"Unable to create new topic described by {description}."
                results = self.find_existing_topic(description, no_topic_response)
                if results == no_topic_response:
                    from plisp.LispTools import make_list
                    return make_list([])
                else:
                    from plisp.LispTools import make_list
                    matches = results.split("|")
                    return make_list([v.string().strip() for v in matches])
            else:
                raise Exception(f"First argument to {fname} must be a string topic description.")

        env.map_function_template(SimpleFunctionTemplate(fname, find_topics))

        fname = "delete-topic-by-meta-role"
        def delete_topic(template, evaluated_args:[Value]):

            if len(evaluated_args) > 0 and evaluated_args[0].is_string():
                meta_role:str = evaluated_args[0].string()
                new = []
                found = False

                if meta_role in self.role_handlers:
                    self.role_handlers.pop(meta_role)

                    for role, (handler, topic, topic_name_short, meta_role) in self.role_handlers.items():
                        if meta_role == role:
                            found = True
                        else:
                            new.append(to_svalue(meta_role))
                    from plisp.LispTools import make_list
                    return make_list(new)
                else:
                    from plisp.Value import NULL_VALUE
                    return NULL_VALUE

            else:
                raise Exception(f"First argument to {fname} must be a string topic description.")

        env.map_function_template(SimpleFunctionTemplate(fname, delete_topic))




    def add_dialog_activity(self, topic:str, meta_role_key = None, topic_name_short = None):
        from .DialogActivity import DialogActivity
        topic_builder_prompt = f"Proposed new dialog topic handler.  The topic is \"{topic}\".  "
        if meta_role_key:
            topic_builder_prompt += f"I've already chosen the meta-role key as something like {meta_role_key}.  Feel free to modify it slightly to make it unique with any clashing meta-role keys if they exist."
        else:
            topic_builder_prompt += "Find an appropriate meta-role key that doesn't clash with the existing ones.  "

        if topic_name_short:
            topic_builder_prompt += f"I've already chosen the topic name short as something like {topic_name_short}.  Feel free to modify it slightly to avoid clashing with existing topic name shorts."
        else:
            topic_builder_prompt += "Find an appropriate topic name short that doesn't clash with any existing ones.  "

        topic_builder_prompt += "Now return your selected meta-role key, topic and topic name short as a pipe delimited string.  "

        command_env = Environment(self.hive_env)
        command_env.map_value("new-topic-spec-prompt", to_svalue(topic_builder_prompt) )
        pre_processed_prompt = build_prompt_from_template(command_env, self.base_hive_topic_creation_rules)
        dialog_history = parse_roles_from_dialog_string(pre_processed_prompt, True)
        response = process_llm_history(dialog_history, llm_name=self.model_name, url=self.url, api_key=self.api_key)
        components = response.split("|")
        if len(components) == 3:
            new_role, topic, topic_name_short = components
            activity = DialogActivity(self, topic,topic_name_short, new_role )
            self.add_meta_role_handler(activity, topic, topic_name_short, new_role)
            return (activity, new_role)
        else:
            return (None, f"Unable to add new topic from \"{topic}\".  Please try to come up with a different topic description")

    def find_existing_topic(self, topic, no_topic_key = None, max_results = None):
        if max_results is None:
            max_results = 5
        if no_topic_key is None:
            no_topic_key = "topic not found"
        search_env = Environment(self.hive_env)
        search_env.map_value("topic-description", to_svalue(topic))
        search_env.map_value("no-match-string", to_svalue(no_topic_key))
        search_env.map_value("num_results", max_results)
        topic_search_prompt = self.base_hive_topic_search_rules
        processed_prompt = build_prompt_from_template(search_env, topic_search_prompt)
        dialog = parse_roles_from_dialog_string(processed_prompt, True)
        response = process_llm_history(dialog, llm_name=self.model_name, url=self.url, api_key=self.api_key)
        return response.split("|")

    def get_delegated_role(self, response):
        return split_role_message(response)[0]

    def get_message_to_delegated_llm(self, response):
        return split_role_message(response)[1]

    def user_chat(self, message:str) -> str:
        self.logger.logDebug(Hive.hive_logging_tag,
                            f"Sending message: {message} from user to {self.prior_role}", Hive.verbose)

        prior = [self.meta_role]
        prior_delegate = [self]
        # Hive tries to answer user's question directly
        initial_response = self.chat("user", message,None)
        delegated_role = self.get_delegated_role(initial_response)
        message_to_delegated_role = self.get_message_to_delegated_llm(initial_response)

        if delegated_role is None:
            self.prior_role = self.meta_role
            return message_to_delegated_role
        delegated_role = delegated_role[:-1]
        role_that_delegated = prior[-1]
        # In: -> delegated_role, message_to_delegated_role, role_that_delegated
        for i in range(self.max_nested_llm_auto_call):
            # Process message with delegated handler
            delegate_handler = self.role_handlers[delegated_role][0]
            response_from_delegate = delegate_handler.chat(role_that_delegated, message_to_delegated_role, prior_delegate[-1].get_environment())

            # This may produce a new delegated role
            next_delegated_role = self.get_delegated_role(response_from_delegate)
            next_message = self.get_message_to_delegated_llm(response_from_delegate)
            if next_delegated_role is None and next_message:
                return next_message
            next_delegated_role = next_delegated_role[:-1]
            next_message_to_delegated_role = self.get_message_to_delegated_llm(response_from_delegate)
            self.logger.logInfo(Hive.hive_logging_tag,
                                f"{role_that_delegated} -> {delegated_role} ==> {next_delegated_role} via: '{message_to_delegated_role}'",
                                Hive.verbose)

            if next_delegated_role is None:
                # AI is talking to user
                # treat as error
                potential_error = self.system_error_handler.set_error_message(next_delegated_role, delegated_role, next_message_to_delegated_role)
                next_delegated_role, next_message_to_delegated_role = split_role_message(potential_error)
                # AI is talking to user
                # This allows the user to resume the conversation in the next call to self.user_chat
                if next_delegated_role is None:
                    self.prior_role = delegated_role
                    return next_message_to_delegated_role
                next_delegated_role = next_delegated_role[:-1]

            # Can only delegate to a different role
            if next_delegated_role == delegated_role:
                # This handler needs to receive a chastisement from the error
                # This is an intervening role from the error system:

                err_role = self.system_error_handler.meta_role
                self.system_error_handler.set_error_message(next_delegated_role, delegated_role, next_message_to_delegated_role)
                # overwrite the original delegate_handler

                corrected_response_from_delegate = self.system_error_handler.chat(err_role, "n/a",None)
                # Now we check the updated
                next_delegated_role = self.get_delegated_role(corrected_response_from_delegate)
                next_message_to_delegated_role = self.get_message_to_delegated_llm(corrected_response_from_delegate)
                role_that_delegated = delegated_role
                if next_delegated_role is None:
                    # AI decided to ask the user for help
                    self.prior_role = delegated_role
                    return next_message_to_delegated_role
                else:
                    next_delegated_role = next_delegated_role[:-1]
            else:
                self.prior_role = delegated_role
                prior.append(delegated_role)
                role_that_delegated = delegated_role

            delegated_role = next_delegated_role
            message_to_delegated_role = next_message_to_delegated_role
            self.hive_env.map_value("prior-role", to_svalue(role_that_delegated))
            # Now loop around and perform the delegation

        self.logger.logWarning(Hive.hive_logging_tag,
                               "Maximum LLM self communication reached.  User must manually approve further communication",
                               Hive.verbose)
        return message_to_delegated_role


    def chat(self, client_meta_role:str, message:str, client_env:Optional[Environment] = None) -> str:
        if client_env is None:
            client_env = self.hive_env
        pre_processed_prompt = build_prompt_from_template(client_env, self.base_prompt)
        base_dialog_input = parse_roles_from_dialog_string(pre_processed_prompt, True)
        dialog = base_dialog_input + self.dialog_experience

        if client_meta_role == self.user_role:
            response = do_multi_shot_llm_query(dialog, message, llm_name=self.model_name, url=self.url, api_key=self.api_key)
        else:
            message = f"{client_meta_role}:" + message
            response = do_multi_shot_llm_query(dialog, message, llm_name=self.model_name, url=self.url,
                                               api_key=self.api_key)
        self.dialog_experience.append(user_prompt_segment(message))
        self.dialog_experience.append(assistant_prompt_segment(response))
        return response
