# Conversational Financial Agent
This project is an agent inspired by https://github.com/czyssrs/ConvFinQA

# Project Setup & Execution Guide

This project leverages UV for dependency management and environment handling, and uses Ollama as a local backend for running large language models (LLMs), specifically Google’s Gemma3.

## Prerequisites
### 1. Install Ollama (for LLM inference)

To run the project we are using Llama3.2 running locally via Ollama, install Ollama and pull down the model:

```commandline
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

Ollama allows you to run LLMs locally with minimal configuration.

### 2. Install UV (for dependency management)

Install UV with the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To learn more about UV’s advanced features, refer to this detailed [guide](https://www.saaspegasus.com/guides/uv-deep-dive/#advanced-usage).

## Running the Project

To execute the project, run:

```bash
uv run test_conv_fin_agent.py
```

This command is equivalent to  the following:

```
1. Finds or downloads a compatible Python version.
2. Creates and configures a virtual environment in .venv/.
3. Installs project dependencies.
4. Activates the virtual environment.
5. Runs test_conv_fin_agent.py.
```

This will regenerate the results.


