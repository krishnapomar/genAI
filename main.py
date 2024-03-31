from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def localllm(prompt_input):
    try:
        model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Tokenizer loaded")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto')
        print("Model loaded")
        pipeline_value = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
        print("Pipeline created")
        local_llm = HuggingFacePipeline(pipeline=pipeline_value)
        print("Local LLM created")
        prompt = PromptTemplate(
            template="Tell me about {prompt_input}",
            input_variables=["prompt_input"]
        )
        print("Prompt created")
        chain = LLMChain(local_llm, prompt_input)
        print("Chain created")
        output = chain.run(prompt)
        print("Output generated")
        return output
    except Exception as e:
        print(e)
        output = "Exception occurred " + str(e)
        return output

@app.get('/')
def init(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request})

@app.post('/api/process')
async def generate_text(request: Request):
    data = await request.json()
    prompt_input = data['input']
    print(prompt_input)
    output = localllm(prompt_input)
    return output
    