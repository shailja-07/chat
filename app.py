from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
api_key = os.getenv('MY_API_KEY')

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
llm = HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=api_key)

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate information in one complete bullet point for the following topic:\n\n{context}")
        ]
    )
    
    
llm_chain = LLMChain(llm=llm, prompt=prompt)


app = FastAPI()


class Data(BaseModel):
    text: str

@app.get("/")
@app.head("/")
def root():
    return {"message": "Welcome!"}

@app.post("/predict")
def predict(data: Data):
    input_data = data.text 
    
    if input_data:
        info_text=llm_chain.run(input_data)
    else:
        print("enter topic")
    return {"information": info_text}  


