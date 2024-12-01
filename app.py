from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_community.llms import HuggingFaceHub


model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

llm = HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token="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")


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

@app.get('/{name}')
def get_name(name: str):
    return {'hello': f'{name}'}

@app.post("/detail")
def detail(data: Data):
    input_data =data.text.strip() 
    
    if input_data:
        info_text=llm_chain.run(input_data)
    return {"information": info_text}  


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)