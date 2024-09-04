from LLMTools import user_prompt_segment, assistant_prompt_segment, get_delimited_text
from LinearLLMDialog import LinearLLMDialog

import json
class LLMPatternMatcher:
    def __init__(self):
        self.history = []

    def extractFields(self, input_pattern, user):
        key_name = "conversation-topic"
        pattern = f"Switch to my conversation about {key_name}"
        user_input = "Switch to my conversation about getting rid of deer"

        prior = [
            "Your job is to do data extraction from strings.  You will be given a pattern string, P, which will combine ordinary text with pattern capture keys.  The pattern capture keys will be delimited by curly braces and will be a description of the type of text you can expect to find in that section of the string.",
            "You will also be given a raw input string, R, from which you will try to extract substrings in the positions corresponding to the pattern capture keys in P.  If you find substrings that fit that pattern then return a json object whose keys are the pattern capture keys from P that had matching text in R.",
            "For example, if the pattern, P is 'Switch to {directory}' and the input, R, is 'Switch to /usr/local/bin' then you should return: '{\"directory\": \"/usr/local/bin\"}'"]

        examples = []
        examples.append(
            user_prompt_segment(f"Suppose P = '{pattern}' and R = '{user_input}' then your response should be:"))
        s = {"directory": "/usr/local/bin"}

        o = json.dumps(s)

        examples.append(assistant_prompt_segment(f"`{o}`"))
        try:
            model = LinearLLMDialog(prior, prior_user_assist_context=examples, temp=0)
            resp = model.chat(f"Suppose P = '{input_pattern}' and R = '{user}' then your response should be:")
            d = get_delimited_text(resp, "`", "`")[0][0]
            out = json.loads(d)
            return out
        except Exception as e:
            return None
