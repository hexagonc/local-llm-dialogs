import random
import unittest
import csv
import numpy

from LLMDialogController import LLMDialogController
from LLMTools import split_role_message, parse_roles_from_dialog_pattern_file
from SupervisorAgent import SupervisorAgent




test_dialog_pattern_file = "tableau_dialog_model_tests.txt"

class MyTestCase(unittest.TestCase):
    def setUp(self):
        print("Starting tests\n\n*******")

    def test_can_create_supervisor_agent(self):
        supervisor = SupervisorAgent()

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


    def test_can_drive_dialog_from_dialog_pattern_file(self):
        auditor_model_name = "deepseek-small"
        deverbose_model = "llama3"

        from AuditorController import AuditorController
        auditor = AuditorController(auditor_model_name, deverbose_model, deverbose_output = True)

        dialog_pattern_context_file = test_dialog_pattern_file
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
        auditor_model_name = "deepseek-small"
        deverbose_model = "llama3"

        from AuditorController import AuditorController
        auditor = AuditorController(auditor_model_name, deverbose_model, deverbose_output = True)

        base_dialog_controller = LLMDialogController()

        dialog_pattern_context_file = test_dialog_pattern_file

        supervisor = SupervisorAgent(auditor)
        success = supervisor.initializeDialog(dialog_pattern_context_file, base_dialog_controller)
        self.assertTrue(success)



if __name__ == '__main__':
    unittest.main()
