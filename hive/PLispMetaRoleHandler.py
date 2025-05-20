from .MetaRoleHandler import  MetaRoleHandler
from .Hive import Hive
from typing import Optional
from plisp.Environment import Environment
from plisp.Value import Value

class PLispMetaRoleHandler(MetaRoleHandler):
    def __init__(self, hive:Hive):
        super().__init__("plisp", "evaluates lisp code that you send using your own Lisp variable environment.", "Send me lisp code and I will evaluate it and send it back to you")
        self.hive = hive

    def chat(self, client_meta_role:str, message:str, client_env:Optional[Environment] = None) -> str:
        if client_env is None:
            client_env = self.hive.hive_env

        response = f"{client_meta_role}: "
        try:
            result:Value = client_env.evaluate_exp(message)
            response += result.serialize()
        except Exception as e:
            response += f"Error processing lisp:\n {e.args[0]}"
        return response


    def add_self_to_hive(self, hive):
        hive.add_meta_role_handler(self, self.topic, "plisp_processor", self.meta_role)
