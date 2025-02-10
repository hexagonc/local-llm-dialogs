from typing import Optional

from LLMChatConfig import CONFIG_MAP
from LLMDialogController import LLMDialogController
from LinearLLMDialog import LinearLLMDialog


class AuditorController:
    def __init__(self, auditor_model):
        self.auditor_model = auditor_model
        model_config = CONFIG_MAP["model-config"]
        model_name, url, api_key = model_config[auditor_model]

        instructions = """
        You are a small but important component within a larger subsystem.  Your output will only be consumed by other computers so only respond with a single number according to the following convention:
        I will present to you a list of N options that will be numbered from 1 to N.  I will then ask you a question and you should respond only with the number
        corresponding to the option that most accurately answers the question given.
        """
        self.auditor = LinearLLMDialog(prior_system_instructions=[instructions.strip()], model_name = model_name, model_url=url, model_api_key=api_key)


    def get_numerated_list_prompt(self, options:list[str],include_header_footer = None, quote_items = None) ->str:

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

    def getClosestSentiment(self, input:str, possible_sentiments:list[str]) -> Optional[str]:
        options_prompt = self.get_numerated_list_prompt(possible_sentiments)
        question = f"""Consider the following input sentence:
__________________
{input}
__________________
Now consider the following list of sentences:
{options_prompt}
Return the number of the option that most closely matches the sentiment or is most similar in meaning to the input sentence.  Return 0 if none of them match.
"""
        response = self.auditor.chat(question, 0.0)
        response_index = int(response)
        if response_index == 0:
            return ""
        else:
            return possible_sentiments[response_index-1]