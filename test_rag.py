import folder_paths
import os

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# from langchain_community.vectorstores import Milvus
from langchain_milvus.vectorstores import Milvus

from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .test_llm import LLaMA3_1_LLM


class test_rag:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "what's your question?"}), 
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "test"

    CATEGORY = "testrag"

    def test(self, text):
        print(type(text))
        file_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI_RAGDemo/test.pdf")
        print(file_path)

        # 初始化pdf 文档加载器
        loader = PyPDFLoader(file_path=file_path)
        # 将pdf中的文档解析为 langchain 的document对象
        documents = loader.load()
        # 将文档拆分为合适的大小
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        model_name = "BAAI/bge-base-zh-v1.5"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction="为这个句子生成表示以用于检索相关文章："
        )

        vectorstore = Milvus.from_documents(
            documents=docs, # 设置保存的文档
            embedding=model, # 设置 embedding model
            collection_name="book", # 设置 集合名称
            drop_old=True,
            connection_args={"uri": "./milvus_demo.db"},# Milvus连接配置
        )
        PROMPT_TEMPLATE = """
        Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <context>
        {context}
        </context>

        <question>
        {question}
        </question>

        The response should be specific and use statistics or numbers when possible.

        Assistant:"""

        # Create a PromptTemplate instance with the defined template and input variables
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        # Convert the vector store to a retriever
        retriever = vectorstore.as_retriever()


        # Define a function to format the retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = LLaMA3_1_LLM(mode_name_or_path = "/share_nfs/hf_models/meta-llama/Meta-Llama-3.1-8B-Instruct")

        # print(llm("你好呀"))

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )


        res = rag_chain.invoke(text)
        print(res)
        print(type(res))
        return (res,)

NODE_CLASS_MAPPINGS = {
    "testRAG": test_rag,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "testRAG": "TestRAG",
}