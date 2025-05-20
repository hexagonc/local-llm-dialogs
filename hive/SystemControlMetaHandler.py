from .MetaRoleHandler import  MetaRoleHandler
from .Hive import Hive
from typing import Optional
from plisp.Environment import Environment
from plisp.Value import Value

class SystemControlMetaHandler(MetaRoleHandler):
    def __init__(self, hive):
        super().__init__("system-error", "notifies LLMs when they have made a mistake and gives hints on how to correct them", "Provides process hints and error messages")
        self.parent_hive = hive
        self.next_response = "Invalid"

    def set_error_message(self, next_delegated_role:str, delegating_meta_role:str, next_message_to_delegated_role:str):
        from LLMTools import split_role_message
        if next_delegated_role == delegating_meta_role:
            # self reference
            self.next_response = f"{delegating_meta_role}: You are sending a message to yourself. This is because the meta-role key you have declared, {next_delegated_role}: is the same as yours.  This is an error since the meta-role key you are sending messages to must be different from yourself.  This means you don't know how to complete the task.  Can you think of any other meta-role handlers from the table above that might be helpful with completing the task?  Think of the topic descriptions.  If you can't think of any that could be of help that you haven't already tried, you should consider asking help from the user."
            return
        elif next_delegated_role is None:
            parts = next_message_to_delegated_role.split("\n")
            delegated_role, new_message = split_role_message(parts[-1])

            if delegated_role and len(new_message) > 0:
                self.next_response = parts[-1]
                return
        self.next_response = f"{next_delegated_role}: {next_message_to_delegated_role}"




    def add_self_to_hive(self, hive):
        hive.add_meta_role_handler(self, self.topic, "system_messages", self.meta_role)

    def queue_next_response(self, response):
        self.next_response = response
    def chat(self, client_meta_role:str, message:str, client_env:Optional[Environment] = None) -> str:

        return self.next_response