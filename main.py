from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from git import Repo
repo_path = "./test_repo"

repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)

loader = GenericLoader.from_filesystem(
    repo_path + "/libs/core/langchain_core/",
    glob="**/*",
    suffixes = [".py"],
    exclude = ["**/non-utf-8-encoding.py"],
    parser = LanguageParser(language=Language.PYTHON, parser_threshold=500),
)

documents = loader.load()
len(documents)

python_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    language=Language.PYTHON
)

texts = python_splitter.split_documents(documents)
len(texts)

os.environ["OPENAI_API_KEY"] = "sua_chave_api"

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
)

llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1000)
prompt = ChatPromptTemplate.from_messages(
    [
        {
            "system", 
            "Você é um revisor de código experiente. Forneça informações detalhadas sobre a revisão do código e sugestões de melhorias baseado no contexto fornecido abaixo: \n\n {context}"
        },
        {"user", "{input}"}
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "Você pode revisar e sugerir melhorias para o código de RunnableBinding"})

print(response['answer'])
