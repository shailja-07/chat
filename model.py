from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
import joblib 

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

llm = HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token="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")



prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate information in one complete bullet point for the following topic:\n\n{context}")
        ]
    )

llm_chain = LLMChain(llm=llm, prompt=prompt)

joblib.dump(llm_chain,'llm.pkl')
