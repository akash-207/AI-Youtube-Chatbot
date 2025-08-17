from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=Gfr50f6ZBvo",
    add_video_info=False,
    language=["en", "id"],
    translation="en",
)
docs=loader.load()
#print(docs[0].page_content[:500])


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
#print(vector_store.index_to_docstore_id)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever.invoke('What is deepmind')

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)

answer = llm.invoke(final_prompt)
print(answer.content)

# Now we are will form a chain in oreder to invoke the llm once

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke('Can you summarize the video')

print(result)






