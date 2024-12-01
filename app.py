from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
import joblib

app = FastAPI()

llm_chain=joblib.load('llm.pkl')

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
    input_data = data.text.strip()
    
    if not input_data:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    try:
        info_text=llm_chain.run(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"information": info_text}  


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)