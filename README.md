# LangChain Weather Assistant

A conversational AI agent built with LangChain, LangGraph, and Ollama that answers general questions and fetches live weather data via a tool-calling workflow. The UI is served through Gradio.

## Features

- **Weather tool** — automatically invoked for weather-related queries (Chennai, London, New York, Tokyo, Paris)
- **General chat** — answers non-weather questions directly without calling any tool
- **Session memory** — optionally remembers conversational context within a session
- **Audit log** — optionally shows whether a tool was called for each response
- **Gradio UI** — browser-based chat interface

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| [Ollama](https://ollama.com) | Latest |
| `llama3.2` model (via Ollama) | — |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Giri1982/langchain_Coding.git
cd langchain_Coding
```

### 2. Install Ollama and pull the model

Download and install Ollama from https://ollama.com, then pull the required model:

```bash
ollama pull llama3.2
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

### 4. Install dependencies

Using `pip`:

```bash
pip install -e .
```

Or using `uv` (faster):

```bash
uv sync
```

### 5. Run the application

```bash
python main.py
```

The Gradio UI will open automatically in your browser at `http://127.0.0.1:7860`.

## Project Structure

```
├── main.py            # Entry point
├── gradio_app.py      # Gradio UI definition
├── chat_service.py    # Agent routing and conversation logic
├── chatmodel.py       # Ollama model configuration
├── weather_tool.py    # LangChain weather tool
├── pyproject.toml     # Project metadata and dependencies
└── .python-version    # Pinned Python version (3.12)
```

## Usage

| Example prompt | What happens |
|---|---|
| `What is the weather in Chennai?` | Calls the weather tool and returns current conditions |
| `What is the weather in Berlin?` | Tool called; returns "data unavailable" for unsupported cities |
| `Tell me a joke` | Answered directly without any tool call |
| `What is my name?` (with memory on) | Recalls name from earlier in the session |

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core agent and chain framework |
| `langchain-ollama` | Ollama model integration |
| `langgraph` | Agent graph and memory checkpointing |
| `gradio` | Browser-based UI |
