## LLM Console Assistant

### Purpose
This project demonstrates how to use off-the-shelf local and remote large language models to assistant in arbitrary tasks including filesystem operations.
### Installation
#### Prerequisites
- **Python**: I haven't tested this extensively but I'm pretty sure any version after Python 3.8 should work for Windows and MacOS.  I haven't tested on Linux yet.
- **LM Studio**: this code relies on [LM Studio](https://lmstudio.ai/) to serve local large language models.  Local model servers must conform to OpenAI's [completions](https://platform.openai.com/docs/api-reference/chat/create), [embedding](https://platform.openai.com/docs/api-reference/embeddings/create) and [models](https://platform.openai.com/docs/api-reference/models/list) apis.
- **Hardware** 
  - At least 16GB of RAM if running local models.  Most of the good local models require at least 4GB of RAM and obviously, you'll need some extra for other applications in memory. 
  - Internet connection if accessing OpenAI endpoints.  If you don't have the [tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer) files, (e.g., tokenizer.json -- this is model dependent) you'll need an internet connection to estimate total token count in the dialog so as to stay below the max context length for the model.
  - At least 16GB of hard drive space if running local models.  This is an estimate since the smallest models of any quality (for example, [Llama3 with 4 bit quantization](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF)) are going to be around 4GB to 5GB and you may want to add higher fidelity versions as well as an embedding model.
  - For best results running local models, you'll want a processor running any type of Apple Silicon (M2+) or a powerful GPU.
#### Setup
1. Clone this repo or download the compressed zip file
2. After all files are extracted, run either *install_windows.bat* if installing on Windows or *install.sh* if running on MacOS or Linux.  Be sure to run `chmod +x install.sh` to make the sure executable and run `source venv/bin/activate` after the script succeeds.
3. Update *config.json*, overriding defaults if necessary and, especially, specifying your OpenAI API key if you intend to access OpenAI's endpoints.
4. Basic Setup in LM Studio
   1. Install the main model for chat completions, LM Studio's Llama-3 8B parameter model with 4 Bit quantization, [Meta-Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf).  This is a good overall model for speed, intelligence and size.
   2. To use some of the optional APIs supported by the Desktop Assistant, you can also install `bartowski/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q6_K-Q8.gguf`, `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` and `akjindal53244/Llama-3.1-Storm-8B-GGUF/Llama-3.1-Storm-8B.Q8_0.gguf` in order to compare llama3 models with different parameter counts and weight precision. 
   3. Go to the "Local Server" tab on the left sidebar to install the embedding model.  The tab icon looks like a bidirectional arrow: '<->'.
   4. In the "Embedding Model Settings" pane, download the "nomic-embed-text-v1.5" embedding model.  Make sure to select the model after downloading it.  This doesn't happen automatically.
5. Go to the Playground tab (page should say "Multi Model Session") and add the model, "Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" make sure the model identifier is set to "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf".  Unload and reload the model in the Playground if it has a different model identifier.
6. Start the local server on the Playground tab
7. From the installation directory, run the appropriate script, install.sh (for linux or Mac) or install_windows.bat for Windows (windows script needs work) 
8. From the installation directory, make sure the virtual environment is activated:
   1. On Windows, run `venv\scripts\Activate.bat`
   2. On Linux or MacOS, run `source venv\bin\activate`
9. Test that everything works by running unit tests:
   1. Linux or MacOS, run: `python3 -m unittest adv_dialog_tests.py`
   2. In a Windows DOS/Powershell terminal, run: `python -m unittest adv_dialog_tests.py`
10. Start a Jupyter server in order to run the Desktop LLM Assistant Jupyter notebook.  This is the main interface for the code.