import os
import datasets
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers.agents import ReactJsonAgent
from langchain_openai import ChatOpenAI
import logging
from RetrieverTool import RetrieverTool
from OpenAIEngine import OpenAIEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def init():

    # Load the knowledge base
    knowledge_base = datasets.load_dataset("TVRRaviteja/Mental-Health-Data", split="train")
    # Convert dataset to Document objects
    source_docs = [
        Document(page_content=doc["text"])
        for doc in knowledge_base
    ]
    
    logger.info(f"Loaded {len(source_docs)} documents from the knowledge base")

    # Initialize the text splitter
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Split documents and remove duplicates
    logger.info("Splitting documents...")
    docs_processed = []
    unique_texts = {}
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)

    logger.info(f"Processed {len(docs_processed)} unique document chunks")

    # Initialize the embedding model
    logger.info("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")


    # Create the vector database
    logger.info("Creating vector database...")

    vectordb = Chroma.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        persist_directory="chroma"
    )

    # Create an instance of the RetrieverTool
    global retriever_tool
    retriever_tool = RetrieverTool(vectordb)

    logger.info("Vector database created successfully")

    llm_engine = OpenAIEngine()
    # Create the agent
    global agent
    agent = ReactJsonAgent(tools=[retriever_tool], llm_engine=llm_engine, max_iterations=3, verbose=2)


# Function to run the agent
def run_agentic_rag(question: str) -> str:
    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
    give a comprehensive answer to the question below.
    Respond only to the question asked, response should be concise and relevant to the question.
    If you cannot find information, do not give up and try calling your retriever again with different arguments!
    Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
    Your queries should not be questions but affirmative form sentences: e.g. rather than "How to check personality scores of someone who is open and agreeable?", query should be "find me personality scores of someone who is open and agreeable".

    Question:
    {question}"""

    return agent.run(enhanced_question)

# Standard RAG function
def run_standard_rag(question: str) -> str:
    prompt = f"""Given the question and supporting documents below, give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.

    Question:
    {question}
    """
    messages = [{"role": "user", "content": prompt}]

    reader_llm = ChatOpenAI(model="gpt-4-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    ai_msg = reader_llm.invoke(messages)

    return ai_msg.content

def main():
    init()
    question = """
    How can i check my score? If I am procarstinating and but at the same time I have imposter syndrome.
    """
    print(f"Question: {question}")

    agentic_answer = run_agentic_rag(question)
    print("Agentic RAG Answer:")
    print(f"Answer: {agentic_answer}")

    standard_answer = run_standard_rag(question)
    print("\nStandard RAG Answer:")
    print(f"Answer: {standard_answer}")


if __name__ == "__main__":
    main()