import random
import unittest
import csv
import numpy

from LLMDialogController import LLMDialogController
from SupervisorAgent import SupervisorAgent


def split_role_message(input):
    import re
    pattern = r"\s*((([\w, \d, _]+)\:+)\**)(.*)"
    match = re.search(pattern, input)
    role = None
    message = input
    if match:
        role = match.group(1)
        message = match.group(4).strip()
        if not message:
            message = None

    return (role, message)

def parse_roles_from_dialog_pattern_file(dialog_pattern_context_file):
    # dialog consists of a list of tuples, (base_role:str, message:list[(bool, str)])
    dialog = []
    previous_role = None
    with open(dialog_pattern_context_file, "r") as w:
        for line in w:
            role, message = split_role_message(line.strip())
            if role:
                base_role = role.strip("*")
                is_preferred = len(role) > len(base_role)
                if previous_role is None:
                    dialog.append((base_role, [(is_preferred, message)]))
                elif previous_role == base_role:
                    all_role_messages = dialog[-1][1]
                    all_role_messages.append((is_preferred, message))
                else:
                    dialog.append((base_role, [(is_preferred, message)]))
                previous_role = base_role
            else:
                all_role_messages = dialog[-1][1]
                message_pref, message_being_updated = all_role_messages[-1]
                all_role_messages[-1] = (message_pref, message_being_updated + "\n" + message)
    return dialog

class MyTestCase(unittest.TestCase):
    def setUp(self):
        print("Starting tests\n\n*******")

    def test_can_create_supervisor_agent(self):
        auditor_model_name = "llama3"
        supervisor = SupervisorAgent(auditor_model_name)

    def test_can_extract_role_message_from_line(self):
        lines = [("user: system: import dialog_filesystem_actions_bash.json", "user:", "system: import dialog_filesystem_actions_bash.json"),
                 ("	system:* confirms that the correct file is being used", "system:*", "confirms that the correct file is being used"),
                 ("resume::", "resume::", None),
                 ("user_1: what time is it?", "user_1:", "what time is it?"),
                 ("Nothing", None, "Nothing"),
                 ("", None, "")]
        for input, exp_role, exp_message in lines:
            role, message = split_role_message(input)
            self.assertTrue((role, message) == (exp_role, exp_message))



    def test_can_parse_roles_from_dialog_file(self):
        dialog_pattern_context_file = "starting_uml_dialog_pattern_context.txt"

        # dialog consists of a list of tuples, (base_role:str, message:list[(bool, str)])
        dialog = parse_roles_from_dialog_pattern_file(dialog_pattern_context_file)
        expected_length = 20
        self.assertTrue(len(dialog) == expected_length)



    def test_can_read_dialog_pattern_file(self):
        dialog_pattern_context_file = "starting_uml_dialog_pattern_context.txt"

        lines = []
        with open(dialog_pattern_context_file, "r") as w:
            for line in w:
                lines.append(line)

        self.assertTrue(len(lines) == 35)

    def test_can_drive_dialog_from_dialog_pattern_file(self):
        auditor_model_name = "llama3"

        from AuditorController import AuditorController
        auditor = AuditorController(auditor_model_name)

        dialog_pattern_context_file = "starting_uml_dialog_pattern_context.txt"
        dialog_pattern_spec = parse_roles_from_dialog_pattern_file(dialog_pattern_context_file)

        dialog_controller = LLMDialogController()
        # The dialog_pattern_spec is a list of role/input pairs

        input_roles = {"user:"}

        response = None

        success = True
        for base_role, messages in dialog_pattern_spec:
            if base_role in input_roles:
                is_preferred, selected_message = random.choice(messages)
                response = dialog_controller.chat(user_input=selected_message, contWithStd=False)
            elif response:
                preferred_response = set()
                response_sentiments = []
                for (preferred_sentiment, sentiment) in messages:
                    response_sentiments.append(sentiment)
                    if preferred_sentiment:
                        preferred_response.add(sentiment)
                if len(response_sentiments) > 1:
                    best_sentiment = auditor.getClosestSentiment(response, response_sentiments)
                    if best_sentiment in preferred_response:
                        continue
                    else:
                        success = False
                        break
        print(f"Succeeded in running dialog pattern: {success}")
        self.assertTrue(success)

    def test_can_load_dialog_pattern(self):
        base_dialog_controller = LLMDialogController()

        auditor_model_name = "llama3"
        dialog_pattern_context_file = "starting_uml_dialog_pattern_context.txt"

        supervisor = SupervisorAgent(auditor_model_name)
        success = supervisor.initializeDialog(dialog_pattern_context_file, base_dialog_controller)
        self.assertTrue(success)



if __name__ == '__main__':
    unittest.main()
