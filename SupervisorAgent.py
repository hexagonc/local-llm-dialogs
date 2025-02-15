import random

from LLMDialogController import LLMDialogController
from LLMTools import parse_roles_from_dialog_pattern_file


class SupervisorAgent:
    def __init__(self, auditor_controller = None, logger = None):
        from AuditorController import AuditorController
        self.logger = logger
        if auditor_controller is None:
            auditor_model_name = "deepseek-small"
            deverbose_model = "llama3"
            self.output_auditor = AuditorController(auditor_model_name, deverbose_model, deverbose_output=True, logger=logger)
        else:
            self.output_auditor = auditor_controller

    def initializeDialog(self, dialog_pattern_context_file, base_dialog_controller = None):
        if base_dialog_controller is None:
            base_dialog_controller = LLMDialogController()
        auditor = self.output_auditor
        dialog_pattern_spec = parse_roles_from_dialog_pattern_file(dialog_pattern_context_file)
        input_roles = {"user:"}

        response = None
        if self.logger:
            self.logger.info(f"Initializing chat with dialog pattern: {dialog_pattern_context_file}")
        success = True
        for base_role, messages in dialog_pattern_spec:

            if base_role in input_roles:

                is_preferred, selected_message = random.choice(messages)
                log_message = f"Base Input: {base_role}: {selected_message}"
                if self.logger:
                    self.logger.info(log_message)
                response = base_dialog_controller.chat(user_input=selected_message, contWithStd=False)
            elif response:
                log_message = f"Agent response: {base_role}: {response}"
                if self.logger:
                    self.logger.info(log_message)
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
        return success
