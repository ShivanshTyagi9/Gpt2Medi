import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document


df = pd.read_csv("Treatments.csv")

df["text"] = df.apply(lambda row: f"Disease: {row['Name']}\nTreatments: {row['Treatments']}", axis=1)
documents = df.to_dict(orient="records")

docs = [Document(page_content=doc["text"], metadata={"disease": doc["Name"]}) for doc in documents]

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./treatments_db")
vectorstore.persist()

print("Vectorstore built and saved to ./treatments_db")