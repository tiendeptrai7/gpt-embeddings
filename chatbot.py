import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import (
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, 
    UnstructuredFileLoader, CSVLoader, MWDumpLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import glob

# Default configuration constants
REQUEST_TIMEOUT_DEFAULT = 10
TEMPERATURE_DEFAULT = 0.0
CHAT_MODEL_NAME_DEFAULT = "gpt-3.5-turbo"
CHUNK_SIZE_DEFAULT = 1000
CHUNK_OVERLAP_DEFAULT = 0
OPENAI_API_VERSION = "2023-05-15"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class DocChatbot:
    llm: ChatOpenAI
    condense_question_llm: ChatOpenAI
    embeddings: OpenAIEmbeddings
    vector_db: FAISS
    chatchain: BaseConversationalRetrievalChain

    def __init__(self) -> None:
        # Initialize LLM and embeddings, without streaming
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key, "API key must be provided in the environment"

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", REQUEST_TIMEOUT_DEFAULT))
        self.temperature = float(os.getenv("TEMPERATURE", TEMPERATURE_DEFAULT))
        self.chat_model_name = os.getenv("CHAT_MODEL_NAME", CHAT_MODEL_NAME_DEFAULT)

        if self.api_key.startswith("sk-"):
            self.init_llm_openai(False)
        else:
            self.init_llm_azure(False)

        self.embeddings = OpenAIEmbeddings()

    def init_llm_openai(self, streaming: bool, condense_question_container=None, answer_container=None) -> None:
        # Initialize LLM for OpenAI API
        try:
            self.llm = ChatOpenAI(
                temperature=self.temperature,
                openai_api_key=self.api_key,
                request_timeout=self.request_timeout,
                model=self.chat_model_name,
                streaming=streaming,
                callbacks=[StreamHandler(answer_container)] if streaming else []
            )

            if streaming:
                self.condense_question_llm = ChatOpenAI(
                    temperature=self.temperature,
                    openai_api_key=self.api_key,
                    request_timeout=self.request_timeout,
                    streaming=True,
                    model=self.chat_model_name,
                    callbacks=[StreamHandler(condense_question_container, "🤔...")]
                )
            else:
                self.condense_question_llm = self.llm

        except Exception as e:
            print(f"Error initializing OpenAI LLM: {e}")

    def init_llm_azure(self, streaming: bool, condense_question_container=None, answer_container=None) -> None:
        # Initialize LLM for Azure OpenAI Service
        try:
            assert os.getenv("OPENAI_GPT_DEPLOYMENT_NAME") and os.getenv("OPENAI_API_BASE")
            assert os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")

            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("OPENAI_GPT_DEPLOYMENT_NAME"),
                temperature=self.temperature,
                openai_api_version=OPENAI_API_VERSION,
                openai_api_type="azure",
                openai_api_base=os.getenv("OPENAI_API_BASE"),
                openai_api_key=self.api_key,
                request_timeout=self.request_timeout,
                streaming=streaming,
                callbacks=[StreamHandler(answer_container)] if streaming else []
            )

            if streaming:
                self.condense_question_llm = AzureChatOpenAI(
                    deployment_name=os.getenv("OPENAI_GPT_DEPLOYMENT_NAME"),
                    temperature=self.temperature,
                    openai_api_version=OPENAI_API_VERSION,
                    openai_api_type="azure",
                    openai_api_base=os.getenv("OPENAI_API_BASE"),
                    openai_api_key=self.api_key,
                    request_timeout=self.request_timeout,
                    model=self.chat_model_name,
                    streaming=True,
                    callbacks=[StreamHandler(condense_question_container, "🤔...")]
                )
            else:
                self.condense_question_llm = self.llm

        except AssertionError as e:
            print("Azure environment variables are missing")
        except Exception as e:
            print(f"Error initializing Azure LLM: {e}")

    def init_streaming(self, condense_question_container, answer_container) -> None:
        # Initialize LLM with streaming support
        if self.api_key.startswith("sk-"):
            self.init_llm_openai(True, condense_question_container, answer_container)
        else:
            self.init_llm_azure(True, condense_question_container, answer_container)

    def init_chatchain(self, chain_type: str = "stuff") -> None:
        # Initialize Conversational Retrieval Chain
        prompt_template = PromptTemplate.from_template("""
            Given the following conversation and a follow-up input, rephrase the standalone question.
            Chat History:
            {chat_history}
            Follow Up Input:
            {question}
            Standalone Question:
        """)

        if not hasattr(self, "vector_db") or not self.vector_db:
            raise ValueError("Vector database must be initialized before using the chat chain.")

        self.chatchain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(),
            chain_type=chain_type,
            condense_question_prompt=prompt_template,
            return_source_documents=True,
            verbose=False,
        )

    def get_answer_with_source(self, query, chat_history):
        result = self.chatchain({
            "question": query,
            "chat_history": chat_history
        }, return_only_outputs=True)
        return result['answer'], result['source_documents']

    def get_answer(self, query, chat_history):
        try:
            chat_history_for_chain = [
                (chat_history[i]['content'], chat_history[i+1].get('content', "")) 
                for i in range(0, len(chat_history), 2)
            ]
            result = self.chatchain({"question": query, "chat_history": chat_history_for_chain})
            return result['answer'], result['source_documents']
        except Exception as e:
            print(f"Error getting answer: {e}")

    def load_vector_db_from_local(self, path: str, index_name: str):
        self.vector_db = FAISS.load_local(path, self.embeddings, index_name)
        print(f"Loaded vector db from local: {path}/{index_name}")

    def save_vector_db_to_local(self, path: str, index_name: str):
        FAISS.save_local(self.vector_db, path, index_name)
        print("Vector db saved to local")

    def init_vector_db_from_documents(self, file_list: List[str]):
        chunk_size = int(os.getenv("CHUNK_SIZE", CHUNK_SIZE_DEFAULT))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", CHUNK_OVERLAP_DEFAULT))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        docs = []
        for file in file_list:
            ext_name = os.path.splitext(file)[-1]
            loader = self.get_loader_for_extension(ext_name, file)
            doc = loader.load_and_split(text_splitter)
            docs.extend(doc)
            print(f"Processed document: {file}")

        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        print("Vector db initialized.")

    @staticmethod
    def get_loader_for_extension(ext_name: str, file: str):
        if ext_name == ".pptx":
            return UnstructuredPowerPointLoader(file)
        elif ext_name == ".docx":
            return UnstructuredWordDocumentLoader(file)
        elif ext_name == ".pdf":
            return PyPDFLoader(file)
        elif ext_name == ".csv":
            return CSVLoader(file_path=file)
        elif ext_name == ".xml":
            return MWDumpLoader(file_path=file, encoding="utf8")
        else:
            return UnstructuredFileLoader(file)

    def get_available_indexes(self, path: str):
        return [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(f"{path}/*.faiss")]
