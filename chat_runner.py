from prompt_toolkit import print_formatted_text




def test_full_screen_yesno_dialog():
    from prompt_toolkit.shortcuts import yes_no_dialog

    result = yes_no_dialog(
        title="Yes/No dialog example", text="Do you want to confirm?"
    ).run()

    print(f"Result = {result}")

def test_fullscreen_input_dialog():
    from prompt_toolkit.shortcuts import input_dialog
    result = input_dialog(
        title="Input dialog example", text="Please type your name:"
    ).run()

    print(f"Result = {result}")

def test_vi_mode():
    from prompt_toolkit import prompt
    print("You have Vi keybindings here. Press [Esc] to go to navigation mode.")
    answer = prompt("Give me some input: ", multiline=False, vi_mode=True)
    print(f"You said: {answer}")

def prompt_continuation(width, line_number, wrap_count):
    """
    The continuation: display line numbers and '->' before soft wraps.

    Notice that we can return any kind of formatted text from here.

    The prompt continuation doesn't have to be the same width as the prompt
    which is displayed before the first line, but in this example we choose to
    align them. The `width` input that we receive here represents the width of
    the prompt.
    """
    from prompt_toolkit.formatted_text import HTML
    if wrap_count > 0:
        return " " * (width - 3) + "-> "
    else:
        text = ("> ").rjust(width)
        return HTML("<strong>%s</strong>") % text

if __name__ == "__main__":
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    our_history = FileHistory(".example-history-file")

    # The history needs to be passed to the `PromptSession`. It can't be passed
    # to the `prompt` call because only one history can be used during a
    # session.
    session = PromptSession(history=our_history)

    import importlib
    import LLMDialogController

    importlib.reload(LLMDialogController)

    ##########################################
    ##  Main Chat Interface Loop
    ## Call run this cell to converse with the default configuration.  This assumes you are running LM Studio in server mode and are
    ## running a LLama3 model with model identifier: lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
    ## You also need to be running the embedding model: nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf

    base_dialog_path = "."
    user_input = f"system: import dialog_filesystem_actions_bash.json"

    print(base_dialog_path)
    dialog_controller = LLMDialogController.LLMDialogController(dialog_index_path=base_dialog_path)
    resp = dialog_controller.chat(user_input, contWithStd=False)

    test_models = "openai"
    command = f"system: use model {test_models}"
    resp = dialog_controller.chat(command, contWithStd=False)
    command = "Display the current date and time from the terminal using the format: \"{Month} {day of month}, {year} {hour of day in 24 hour scale}:{minutes in hour}\""
    resp = dialog_controller.chat(command, contWithStd=False)
    command = "system: run shell command"
    print("Press [Esc] followed by [Enter] to accept input.")
    resp = dialog_controller.chat(command, contWithStd=True)





