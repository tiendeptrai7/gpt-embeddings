# chat-with-your-doc

`chat-with-your-doc` is a demonstration application that leverages the capabilities of ChatGPT/GPT-4 and LangChain to enable users to chat with their documents. This repository hosts the codebase, instructions, and resources needed to set up and run the application.

## Introduction

The primary goal of this project is to simplify the interaction with documents and extract valuable information with using natural language. This project is built using LangChain and GPT-4/ChatGPT to deliver a smooth and natural conversational experience to the user, with support for both `Azure OpenAI Services` and `OpenAI`

![](static/web_ui.png)

## Updates

- 20230709: Add Support for OpenAI API
- 20230703: Web UI changed to Streamlit, with support for streaming

## Features

- Upload documents as external knowledge base for GPT-4/ChatGPT, support both `Azure OpenAI Services` and `OpenAI`
- Support various format including PDF, DOCX, PPTX, TXT and etc.
- Chat with the document content, ask questions, and get relevant answers based on the context.
- User-friendly interface to ensure seamless interaction.

### Todo
- [ x ] Show source documents for answers in the web gui
- [ x ] Support streaming of answers
- [ ] Support swith of chain type and streaming LangChain output in the web gui

## Architecture

![](./static/architecture.png)

## Installation

> Suggest to install on Ubuntu instead of CentOS/Debian. See Issue https://github.com/linjungz/chat-with-your-doc/issues/12

To get started with `Chat-with-your-doc`, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/linjungz/chat-with-your-doc.git
```

2. Change into the `chat-with-your-doc` directory:

```bash
cd chat-with-your-doc
```

3. Install the required Python packages:

Create virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install depenancies:

```bash
pip install -r requirements.txt
```

## Configuration

> In this project we're supporting both API from OpenAI and Azure OpenAI Service. There're some environmnet variables that are common for the two APIs while some are unique. The following table lists all the env vars that're supported:

| Environment Variables | Azure OpenAI Service | OpenAI |
| --- | --- | --- |
| OPENAI_API_BASE | :white_check_mark: | |
| OPENAI_API_KEY  | :white_check_mark: | :white_check_mark: |
| OPENAI_GPT_DEPLOYMENT_NAME | :white_check_mark: | |
| OPENAI_EMBEDDING_DEPLOYMENT_NAME | :white_check_mark: | :white_check_mark: |
| CHAT_MODEL_NAME | | :white_check_mark: |
| REQUEST_TIMEOUT | :white_check_mark: | :white_check_mark: |
| VECTORDB_PATH | :white_check_mark: | :white_check_mark: |
| TEMPERATURE | :white_check_mark: | :white_check_mark: |
| CHUNK_SIZE | :white_check_mark: | :white_check_mark: |
| CHUNK_OVERLAP | :white_check_mark: | :white_check_mark: |


### Azure OpenAI Services

1. Obtain your Azure OpenAI API key, Endpoint and Deployment Name from the [Azure Portal](https://portal.azure.com/).
2. Create `.env` in the root dir and set the environment variables in the file:

```
OPENAI_API_BASE=https://your-endpoint.openai.azure.com
OPENAI_API_KEY=your-key-here
OPENAI_GPT_DEPLOYMENT_NAME=your-gpt-deployment-name
OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name
```
Here's where you can find the deployment names for GPT and Embedding:
![Alt text](./static/deployment.png)

### OpenAI

1. Obtain your OpenAI API key from the [platform.openai.com](https://platform.openai.com/account/api-keys).
2. Create `.env` in the root dir and set the environment variable in the file:

```
OPENAI_API_KEY=your-key-here
CHAT_MODEL_NAME="gpt-4-0314"
```

## Usage: Web

This will initialize the application based on `Streamlit` and open up the user interface in your default web browser. You can now upload a document to create a knowledge base and start a conversation with it.

```bash
$ streamlit run chat_web_st.py --server.address '0.0.0.0'

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.


  You can now view your Streamlit app in your browser.

  URL: http://0.0.0.0:8501```
```

Note that the previous Web UI built using Gradio is deprecated and no longer maintained. You could find the code in the [chat_web.py](chat_web.py) file.

## Usage: CLI

The CLI application is built to support both `ingest` and `chat` commands. Python library `typer` is used to build the command line interface.

### **Ingest**

This command would take the documents as input, split the texts, generate the embeddings and store in a vector store `FAISS`. The vector store would be store locally for later used for chat.

![](./static/cli_ingest.png)

For example if you want to put all the PDFs in the directory into one single vector store named `surface`, you could run:
    
```bash
$ python chat_cli.py ingest --path "./data/source_documents/*.pdf" --name surface
```
Note that the path should be enclosed with double quotes to avoid shell expansion.

### **Chat**

This command would start a interactive chat, with documents as a external knowledge base in a vector store. You could choose which knowledge base to load for chat. 

![CLI Chat](./static/cli_chat.png)

Two sample documents about Surface has been provided in the [data/source_document](data/source_documents) directory and already ingested into the default vector store `index`, stored in the [data/vector_store](data/vector_store). You could run the following command to start a chat with the documents:

```bash
$ python chat_cli.py chat
```

Or you could specify the vector store to load for chat:

```bash
$ python chat_cli.py chat --name surface
```

## Reference

`Langchain` is leveraged to quickly build a workflow interacting with Azure GPT-4. `ConversationalRetrievalChain` is used in this particular use case to support chat history. You may refer to this [link](https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html) for more detail.

For `chaintype`, by default `stuff` is used. For more detail, please refer to this [link](https://docs.langchain.com/docs/components/chains/index_related_chains)

## Credits

- The LangChain usage is inspired by [gpt4-pdf-chatbot-langchain](https://github.com/mayooear/gpt4-pdf-chatbot-langchain)
- The integration of langchain streaming and Stremlit is inspired by [Examples from Streamlit](https://github.com/streamlit/llm-examples)
- The processing of documents is inspired by [OpenAIEnterpriseChatBotAndQA](https://github.com/RicZhou-MS/OpenAIEnterpriseChatBotAndQA)

## License

`chat-with-your-doc` is released under the [MIT License](LICENSE). See the `LICENSE` file for more details.
