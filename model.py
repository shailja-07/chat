from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
import joblib 

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
hf_api_token = "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"

llm = HuggingFaceEndpoint(
    model=model_name,
    token=hf_api_token  
)

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate information in one complete bullet point for the following topic:\n\n{context}")
        ]
    )

llm_chain = LLMChain(llm=llm, prompt=prompt)

joblib.dump(llm_chain,'llm.pkl')
