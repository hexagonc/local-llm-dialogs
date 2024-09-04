import os
from LLMTools import serialize_to_file, deserialize_from_file
class LinearLLMDialogRecorder:
    def __init__(self, dialog_file, prior_context = None, auto_save_on_update:bool = None):
        if os.path.exists(dialog_file):
            self.history = deserialize_from_file(dialog_file)
        else:
            self.history = []

        if prior_context:
            self.history+=prior_context
        self.dialog_file = dialog_file
        if auto_save_on_update is None:
            auto_save_on_update = True
        self.auto_save_on_update = auto_save_on_update

    def should_save_on_update(self, status:bool):
        self.auto_save_on_update = status
        return status

    def update_history(self, dialog_segment):
        self.history.append(dialog_segment)
        if self.auto_save_on_update:
            self.save()


    def get_dialog_listener(self):
        return lambda seg: self.update_history(seg)


    def save(self, output_file = None):
        if output_file is None:
            output_file = self.dialog_file
        if os.path.exists(output_file):
            os.remove(output_file)
        serialize_to_file(self.history, output_file, True)

    def get_dialog(self):
        return self.history.copy()
    def load_dialog(self, input_file):
        self.dialog_file = input_file
        self.history = deserialize_from_file(input_file)
        return self.history