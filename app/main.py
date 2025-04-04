from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

model_path = "./Gpt2Medi"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

class Query(BaseModel):
    disease: str

@app.post("/predict")
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
