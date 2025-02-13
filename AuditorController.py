from typing import Optional

from LLMChatConfig import CONFIG_MAP
from LinearLLMDialog import LinearLLMDialog
from LLMTools import get_numerated_list_prompt

class AuditorController:
    def __init__(self, auditor_model, deverbose_model = None, deverbose_output = None):
        self.auditor_model = auditor_model
        model_config = CONFIG_MAP["model-config"]
        model_name, url, api_key = model_config[auditor_model]
        self.deverbose_output = deverbose_output
        instructions = """
        You are a small but important component within a larger subsystem.  Your output will only be consumed by other computers so only respond with a single number according to the following convention:
        I will present to you a list of N options that will be numbered from 1 to N.  I will then provide a sentence and you should respond only with the number
        corresponding to the option that most accurately matches that sentence in sentiment and meaning.  There may not be an exact match but you must return the option that is most similar.  Only return 
        0 if there is truly no good answer and the only response would be a random guess.
        """
        self.auditor_sys_prompt = instructions
        self.auditor = LinearLLMDialog(prior_system_instructions=[instructions.strip()], model_name = model_name, model_url=url, model_api_key=api_key)

        deverbose_instructions = """
        You are a small but important component within a larger subsystem.  Return only the final answer as succinctly and accurately as possible.
        """

        if deverbose_model is None:
            deverbose_model = auditor_model
        model_name, url, api_key = model_config[deverbose_model]

        self.deverboseHelper = LinearLLMDialog(prior_system_instructions=[deverbose_instructions.strip()], model_name = model_name, model_url=url, model_api_key=api_key)



    def getClosestSentiment(self, input:str, possible_sentiments:list[str]) -> Optional[str]:
        options_prompt = get_numerated_list_prompt(possible_sentiments)
        question = f"""Consider the following input sentence:
__________________
{input}
__________________
Now consider the following list of sentences:
{options_prompt}
Return the number of the option that most closely matches the sentiment or is most similar in meaning to the input sentence.  Return 0 if none of them match.
"""
        response = self.auditor.chat(question, 0.0)

        if self.deverbose_output:
            response = self.deverbose(response)
        response_index = int(response)
        if response_index == 0:
            return ""
        else:
            return possible_sentiments[response_index-1]

    def deverbose(self, verbose_output, requested_output_format = None):
        if requested_output_format is None:
            output_formatting = self.auditor_sys_prompt

        prompt = f"""Consider the following sentence:
------------
${verbose_output}
------------
This sentence is supposed to adhere to the following rules in order to facilitate data processing:
------------
${output_formatting}
------------
Express this output according to the formatting rules provided as concisely as possible.
"""
        resp = self.deverboseHelper.chat(prompt, 0.0)
        return resp