from typing import Optional
from plisp.Environment import Environment

class MetaRoleHandler:
    def __init__(self, meta_role:str, topic:str, on_connect_message:Optional[str] = None):
        self.meta_role = meta_role
        self.topic = topic
        if on_connect_message:
            self.on_connect_message = on_connect_message
        else:
            self.on_connect_message = f"Specialized in the topic of {topic}."

    def chat(self, client_meta_role:str, message:str, client_env:Optional[Environment] = None) -> str:
        return ""

    def queue_next_response(self, response):
        pass

    def get_environment(self):
        return Environment()

    def add_self_to_hive(self, hive):
        hive.add_meta_role_handler(self, self.topic, "plisp_processor", self.meta_role)

