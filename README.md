## LLM Console Assistant

### Purpose
This project demonstrates how to use off-the-shelf local and remote large language models to assistant in arbitrary tasks including filesystem operations.
### Installation
#### Prerequisites
##### Python
I haven't tested this extensively but I'm pretty sure any version after Python 3.8 should work for Windows and MacOS.  I haven't tested on Linux yet.
##### Configuring LLM Server
Any model server that exposes a REST API compatible with the `completions`, `embeddings` and `models` endpoints from OpenAI can be accessed.  These models are configured in the
`config.json` that you create by renaming [config.json.template](https://github.com/hexagonc/local-llm-dialogs/blob/main/config.json.template).   The main dialog management REPL can be run from either a terminal shell or a Jupyter notebook.  [ollama](https://github.com/ollama/ollama/blob/main/README.md#quickstart) is a good option for a lightweight configuration that can be installed remotely via terminal commands.  [LM Studio](https://lmstudio.ai/) is a good option for users that prefer a graphical user interface for managing their local models.
Here is an example `config.json` file:
```
{
	"default-model-name": "llama3:8b-instruct-q4_K_M",
	"default-model-api-url": "http://localhost:11434/v1/",
	"default-model-api-key": "ollama",
	"default-embedding-model-name": "mxbai-embed-large:latest",
	"default-embedding-model-api-key": "ollama",
	"default-embedding-model-url": "http://localhost:11434/v1/",
	"model-config": {
		"llama3": ["llama3:8b-instruct-q4_K_M","http://localhost:11434/v1/", "ollama"],
		"deepseek-small": ["deepseek-r1:latest", "http://localhost:11434/v1/", "ollama"],
		"deepseek-large": ["deepseek-r1:14b", "http://localhost:11434/v1/", "ollama"],
		"openai": ["gpt-4o", "https://api.openai.com/v1", "your openai api key"]
	}
}
```
The key "model-config" describes the local and remote models that the desktop assistant is allowed to connect to.  The keys of the "model-config" object are the short names to 
be used when referring to the model.  The value of each key specifies the configuration options for each named model as a list \[{model configured name in LLM server}, {LLM server API endpoint} {LLM API server API key}\].


The keys "default-model-name", "default-model-api-url" and "default-model-api-key" present the configuration for an worker model that is used to assist with internal processing.  This model should be as small as possible while 
be intelligent enough to pass the tests in `assess_default_models.sh`. (TBD)  [Llama3 instruct](https://ollama.com/library/llama3:8b-instruct-q4_K_M) with 8B parameters and 4_K_M quantization works well as a worker but is over 4GB large and 
smaller ones may be just as effective as the technology improves.  

In addition, the desktop assistant also uses an embedding model to help with meta-command processing.  This is configured by the keys, "default-embedding-model-name", "default-embedding-model-api-key" and "default-embedding-model-url".
Again, this model should be as small as necessary to run the tests `asssess_embedding_model.sh`. (e.g., run a command like: `python assess_embedding_models.py --models "all-minilm:latest" "mxbai-embed-large:latest" --url="http://localhost:11434/v1/" --api_key="ollama"`, more documentation TBD)  A good choice here is "mxbai-embed-large" from mixbread.ai: https://ollama.com/library/mxbai-embed-large.

  
name local large language models.  Local model servers must conform to OpenAI's [completions](https://platform.openai.com/docs/api-reference/chat/create), [embedding](https://platform.openai.com/docs/api-reference/embeddings/create) and [models](https://platform.openai.com/docs/api-reference/models/list) apis.
##### Hardware** 
  - At least 16GB of RAM if running local models.  Most of the good local models require at least 4GB of RAM and obviously, you'll need some extra for other applications in memory. 
  - Internet connection if accessing OpenAI endpoints.
  - At least 16GB of hard drive space if running local models.  This is an estimate since the smallest models of any quality (for example, [Llama3 with 4 bit quantization](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF)) are going to be around 4GB to 5GB and you may want to add higher fidelity versions as well as an embedding model.
  - For best results running local models, you'll want a processor running any type of Apple Silicon (M2+) or a powerful GPU.
#### Setup
1. Clone this repo or download the compressed zip file
2. After all files are extracted, run either *install_windows.bat* if installing on Windows or *install.sh* if running on MacOS or Linux.  Be sure to run `chmod +x install.sh` to make the sure executable and run `source venv/bin/activate` after the script succeeds.
3. Create *config.json* from *config.json.template*, overriding defaults if necessary and, especially, specifying your OpenAI API key if you intend to access OpenAI's endpoints.

##### If using LM Studio for the LLM Server:
1. Basic Setup in LM Studio
   1. Install the default model for internal processing as defined in your configuration file, `config.json`.  LM Studio's Llama-3 8B parameter model with 4 Bit quantization, [Meta-Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf) works well for this.  This is a good overall model for speed, intelligence and size.
   2. Install any other local models defined in "model-config" section of `config.json`.  The following models work well when run locally on modest hardware: `bartowski/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q6_K-Q8.gguf`, `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` and `akjindal53244/Llama-3.1-Storm-8B-GGUF/Llama-3.1-Storm-8B.Q8_0.gguf` in order to compare llama3 models with different parameter counts and weight precision. 
   3. Go to the "Local Server" tab on the left sidebar to install the embedding model.  The tab icon looks like a bidirectional arrow: '<->'.
   4. In the "Embedding Model Settings" pane, download the "nomic-embed-text-v1.5" embedding model.  Make sure to select the model after downloading it.  This doesn't happen automatically.
2. Go to the Playground tab (page should say "Multi Model Session") and click "Go".  
3. On this page, you will be given the option to load multiple models into memory.  You will need to have enough RAM to run at least the default and embedding models for minimum desktop assistant functionality.  Use caution when loading additonal models simultaneously as loading more models than you have RAM capacity may cause LM Studio and maybe even your whole computer to freeze.  As you load each model into memory, you will be able to define the models name when specified by the REST API.  Make sure the model name configured here is the same as configured in `config.json`.
4. Start the local server on the Playground tab

##### If using Ollama
1. Make sure the ollama server is running, this is usually on http://localhost:11434/v1/.  Run `ollama serve` if necessary to start the server.
2. Make sure all models referenced in `config.json` have been downloaded to your machine via `ollama list`.  That also shows the model names to be used in the config file.
3. By default, ollama doesn't keep models loaded into memory indefinitely to free up memory.  Instead, it tends to load models into memory as needed and keep them there so long as they are being accessed.  However, after about 5 minutes of idle time, it will automatically unload models.  In order to keep a model loaded without
having to pay the high startup cost when a model has to be reloaded into memory, you can either periodically issue a command to that model or you can reconfigure the model unload timeout to be a long period.  Note that allowing a model to remain in memory for large amounts of time will put you at risk of out-of-memory errors with ollama that could
crash your computer.

#### Main installation
1. From the installation directory, make sure the virtual environment is activated:
   1. On Windows, run `venv\scripts\Activate.bat`
   2. On Linux or MacOS, run `source venv\bin\activate`
2. Test that everything works by running unit tests:
   1. Linux or MacOS, run: `python3 -m unittest adv_dialog_tests.py`
   2. In a Windows DOS/Powershell terminal, run: `python -m unittest adv_dialog_tests.py`
3. Start a Jupyter server in order to run the Desktop LLM Assistant Jupyter notebook, `LLMProjectDialog.ipynb`.  This is the main interface for the code and contains additional usage instructions.
4. Run the desktop assistant REPL from the command: `./run_chat.sh`.  Use the command `./run_chat.sh --help` for basic usage.

### Using the desktop assistant REPL
The Jupyter notebook as well as the `run_chat.sh` console app implements an LLM chat client as an REPL (Read-Eval-Print-Loop from Lisp parlance).  The user can communicate with two different agent roles via this chat interface: the "assistant" role and the "system" role.  All text typed into the chat client are directed to the 
"assistant" agent role by default.  This will direct the user input to the active LLM chat model.  Input for the "system" agent role must be prefixed by the "system:" prefix.  All remaining input will be sent to a separate internal 
"system" agent which is used to assisted in processing system commands to modify the global behavior of the chat client, including changing the active chat assistant model as well as performing prompt engineering functions such as 
rewriting dialog history, impersonating different agent roles, as well as taking special privileged actions such as directly executing shell commands in the current working directory.  See full command reference below.
## Command Reference
Contains a combination of commands to the system as well as commands to the assistant.

### System commands:
##### Switching to a different dialog branch:
`system: switch to branch {new_dialog_branch_filename.json}`
The `new_dialog_branch_filename` should be the absolute path to a file that 

##### Importing the history from a different dialog branch:
`system: import branch {existing_dialog_filename.json}`
If `{existing_dialog_filename.json}` doesn't exist then an error will be raised

##### Popping dialog
`system: pop`

##### Evaluating assistant shell commands
`system: run shell command [(command index)]`
Runs the nth assistant shell command from assistant's output.  Shell commands are assistant output delimited by /* */.  Those commands
will be run a shell and results will be automatically communicated to the assistant.  The system will impersonate the user and return one of:
```
stdout:
{stdout result from shell command}
```
or 
```
stderr:
{stderr result from shell command}
```
or if the shell commands has neither a stdout nor stderr then just return the returncode:
```
returncode: {code}
```

##### Replay dialog history
`system: replay [n]`
Displays the last *n* steps of the current dialog.  This function is useful for providing context in the event that you need to refresh your memory for what you were talking about with the LLM when returning to the dialog from another session (which won't have your dialog history by default).

##### Introduce a file and its contents into the assistant's awareness
`system: introduce the file: {full file path}`

##### Show the content the assistant intends to be written to the current file that has been introduced
`system: show assistant content`


##### Write the assistant's content to a file
`system: write assistant content to {name of output file}`

##### Dynamically changing assistant model
These models are defined in the config.json.  The names you use here are defined in the "model-config" section of the file.
`system: use model name [llama3 | openai | ... {model key defined in "model-config"}]`

## Example dialogs in REPL 
The following chat transcripts provide different usage examples for the desktop assistant REPL.  These dialogs can be reproduced on your own computer for similar effect.
In the examples below, you can also see all three roles at play: the "user" role which is conveyed by the input typed into the REPL, the "assistant" role which is the active large language model 
configured for the REPL and the "system" role which is the REPL itself.  You can think of these three roles as having three separate responsibilities similar
to branches of government.  For most of the conversation, the user and assistant can collaborate to achieve a task.  During the course of the conversation, the large 
language model will propose a shell command.  In this case, the assistant has the role of the legislative branch proposing something to be done (i.g., a "law").  The user looks at the
proposal and decides if that shell command is safe or not, acting as a "judge" of the proposal.  And finally, if the user deems the command as "legal", the "system" role performs a 
role like the executive branch, actually running the command proposal and reporting on the results to the user and assistant.  This forces the LLM to only execute one action at a
time, limiting the damage potential from any mistakes in the actions chosen.

**Transcript of user instructing ChatGPT to use the bash shell to execute a simple command**
The desktop assistant REPL provides an interface where ChatGPT can execute arbitrary commands via standard Python shell utilities.  Use caution when executing shell commands
proposed by ChatGPT or any other model you are chatting with.  It goes without saying that running a command that you don't understand can result in the loss of data
or worse.  Nevertheless, in my anecdotal experience, GPT-4o has a surprisingly comprehensive understanding of the bash shell interface and gives credible 
advice when you ask it how to do something commandline tools.  The first dialog is a "hello, world!" example of conversing with a large language model assistant in order to 
do a simple task on the commandline.  In this case, the user asks ChatGPT to display the current server time using bash commands.
```
user: system: import dialog_filesystem_actions_bash.json

system: Using the dialog file dialog_filesystem_actions_bash.json

user: system: use model openai

system: Changes default model to: gpt-4o

user: Display the current date and time from the terminal using the format: "{Month} {day of month}, {year} {hour of day in 24 hour scale}:{minutes in hour}"

assistant: To display the current date and time in the specified format, you can use the `date` command with a custom format string. Please execute the following command:

\`\`\`sh
/*date +"%B %d, %Y %H:%M"*/
\`\`\`

Press [Esc] followed by [Enter] to accept input.
user: system: run shell command

user: stdout:
January 28, 2025 08:55


assistant: Great! You've successfully displayed the current date and time in the desired format. If you need further assistance, feel free to ask!
```

**Demonstration of desktop assistant ability to execute complex commands given indirectly**
This example shows the ability of the desktop assistant using GPT-4o to extract information intended for human consumption in order to achieve a goal with minimal guidance from the 
user.  The desktop assistant REPL is running in a folder with a [PlantUML](https://plantuml.com/) [activity diagram uml file](https://plantuml.com/activity-diagram-legacy) containing instructions for how to start the PlantUML java app in such as way that the 
UI doesn't block the current terminal session.  The user tells ChatGPT to start up the PlantUML by following the instructions that are in a PlantUML activity diagram.
```
user: system: import dialog_filesystem_actions_bash.json
system: Using the dialog file dialog_filesystem_actions_bash.json
user: system: use model openai
system: Changes default model to: gpt-4o
user: List all files in the current directory

assistant: Please execute the following command to list all files in the current directory:

\`\`\`sh
/*ls -1*/
\`\`\`

user: system: run shell command

user: stdout:
LICENSE
LLMDialogController.py
LLMPatternMatcher.py
LLMProjectDialog.ipynb
LLMTools.py
LinearLLMDialog.py
LinearLLMDialogRecorder.py
README.md
SpeechHashMap.py
__init__.py
__pycache__
adv_dialog_tests.py
anki_flashcards.csv
anki_numpy_cards_mindmap.puml
assess_default_models.py
assess_embedding_models.py
chat_runner.py
check_requirements.py
config.json
config.json.template
dialog_filesystem_actions.json
dialog_filesystem_actions_DOS.json
dialog_filesystem_actions_bash.json
dialog_fs_actions_viewer_and_copying.json
dialog_index.json
filesystem_actions.json
help.txt
install.sh
install_windows.bat
requirements.txt
run_chat.sh
setting_up_plantuml.png
setting_up_plantuml.puml
setup.py
venv


assistant: Here is the list of all files and directories in the current directory. If you need any specific action on these files or directories, please let me know!

system: Processed command: system: run shell command

user: Follow the instructions in setting_up_plantuml.puml for starting PlantUML with bash

assistant: To follow the instructions in `setting_up_plantuml.puml`, we first need to view the contents of the file to understand the steps required to start PlantUML with bash. Please execute the following command to display the contents of the file:

\`\`\`sh
/*cat setting_up_plantuml.puml*/
\`\`\`

user: system: run shell command

user: stdout:
@startuml

start
:Are you using bash or DOS?;
if (bash) then (yes)
  :run the command 'nohup java -DPLANTUML_LIMIT_SIZE=8192 -jar plantuml-1.2024.6.jar -gui > /dev/null 2>&1 &' in the terminal;
else (no)
  :run the command 'start /B java -jar plantuml-1.2024.6.jar -gui';
endif
:Find the Java window of the PlantUML gui;
:Now create a new file in the same directory with extension .puml;
:save your puml code to the file;
:Click on the filename in the PlantUML gui;
:Find the window with the png file of the diagram that gets created;
stop

@enduml

assistant: Based on the instructions from `setting_up_plantuml.puml`, you can start PlantUML with bash by executing the following command:

\`\`\`sh
/*nohup java -DPLANTUML_LIMIT_SIZE=8192 -jar plantuml-1.2024.6.jar -gui > /dev/null 2>&1 &*/
\`\`\`

This command will start the PlantUML GUI in the background. After running this command, look for the Java window of the PlantUML GUI and proceed with creating and saving your `.puml` file as described in the diagram. If you encounter any issues or need further assistance, feel free to ask!

system: Processed command: system: run shell command

user: system: run shell command

user: returncode:0

assistant: Great! PlantUML should now be running in the background with its GUI. You can proceed with creating a new `.puml` file in the same directory, add your PlantUML code to it, and then open the file in the PlantUML GUI to view your diagram. If you have any questions or need further help, just let me know!
system: Processed command: system: run shell command
```
