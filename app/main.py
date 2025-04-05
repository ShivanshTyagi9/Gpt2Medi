from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

app = FastAPI()

model_path = "./Gpt2Medi"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

persist_dir = "./treatments_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)


class Query(BaseModel):
    disease: str

@app.post("/predict/symptoms")
def generate_symptoms(query: Query):
    input_text = f"Symptoms of {query.disease} are:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    output = model.generate(
        input_ids=input_ids,
        max_length=50,
        num_beams=5,
        early_stopping=True,
        repetition_penalty=1.1
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": generated_text}


@app.post("/predict/treatments")
def get_treatments(query: Query):
    results = vectorstore.similarity_search(query.disease, k=1)
    if results:
        treatment_info = results[0].page_content.split("Treatments:")[-1].strip()
        disease_name = results[0].metadata.get("disease", "Unknown")
    else:
        treatment_info = "No treatment found."
        disease_name = "Unknown"

    return {
        "disease": disease_name,
        "treatments": treatment_info
    }