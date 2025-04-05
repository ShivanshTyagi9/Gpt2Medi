FROM python:3.10-slim

ENV TRANSFORMERS_CACHE=/tmp/hf_cache

RUN apt-get update && apt-get install -y git

WORKDIR /code

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY treatments_db ./treatments_db
COPY . .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]