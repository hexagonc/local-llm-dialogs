import os.path

from LLMTools import DEFAULT_API_URL, DEFAULT_API_KEY, LLAMA_LLM_NAME, LM_STUDIO_API_KEY, DEFAULT_DIALOG_TEMP
from LLMTools import merge_dialog_roles, system_prompt_segment, user_prompt_segment, assistant_prompt_segment
from LLMTools import do_multi_shot_llm_query

from LinearLLMDialogRecorder import LinearLLMDialogRecorder


class LinearLLMDialog:
    def __init__(self, prior_system_instructions=None, dialog_listener= None, model_url = None, model_name=None, model_api_key=None, temp = None, prior_user_assist_context = None):
        if prior_system_instructions is None:
            prior_system_instructions = ["""You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."""]
        self.prior_system_instructions = prior_system_instructions
        self.dialog_listener = dialog_listener
        if model_url is None:
            model_url = DEFAULT_API_URL
        if model_name is None:
            model_name = LLAMA_LLM_NAME
        if model_api_key is None:
            model_api_key = LM_STUDIO_API_KEY
        self.model_url = model_url
        self.model_name = model_name
        self.model_api_key = model_api_key
        if temp is None:
            temp = DEFAULT_DIALOG_TEMP
        self.temp = temp

        system_instruction = merge_dialog_roles([system_prompt_segment(_) for _ in prior_system_instructions], role="system")
        if prior_user_assist_context is None:
            prior_user_assist_context = []
        self.dialog_history = system_instruction + prior_user_assist_context
        if dialog_listener is not None:
            dialog_listener(system_instruction[0])

    def chat(self, user_input, temp = None):
        if temp is None:
            temp = self.temp

        if self.dialog_listener:
            self.dialog_listener(user_prompt_segment(user_input))
        response = do_multi_shot_llm_query(self.dialog_history, query=user_input, url = self.model_url, api_key=self.model_api_key, llm_name=self.model_name, temperature=temp)
        if self.dialog_listener:
            self.dialog_listener(assistant_prompt_segment(response))
        return response

    def trimDialog(self, steps):
        self.dialog_history = self.dialog_history[:-steps]

    def startDialogBranchRecording(self, dialog_file, prefill_with_prior_history = None, update = None):
        if update is None:
            update = False

        if prefill_with_prior_history is None:
            prefill_with_prior_history = True
        if os.path.exists(dialog_file):
            recorder = LinearLLMDialogRecorder(dialog_file, auto_save_on_update=True)
            self.dialog_history = recorder.get_dialog()
        else:
            if prefill_with_prior_history:
                recorder = LinearLLMDialogRecorder(dialog_file, self.dialog_history, auto_save_on_update=True)
            else:
                recorder = LinearLLMDialogRecorder(dialog_file, auto_save_on_update=True)
            recorder.save()
        self.dialog_listener = recorder.get_dialog_listener()

    def loadDialog(self, dialog_file):
        """
        Creates switches dialog history to match [dialog_file]
        """
        recorder = LinearLLMDialogRecorder(dialog_file)
        self.dialog_history = recorder.get_dialog()