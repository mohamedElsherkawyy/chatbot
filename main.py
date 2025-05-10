import json
import os
import warnings
import re
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings('ignore')
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key not found in environment variables")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
memory = ConversationBufferMemory()

chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=groq_api_key
)

conversation = ConversationChain(llm=chat, memory=memory)

class ChatResponse(BaseModel):
    message: str

style = """polite tone that speaks in English, 
keep the questions direct and concise, asking only for the required details without adding unnecessary conversation."""
system_message = """
You are a specialized medical and fitness assistant.

When you start the conversation, make sure that you say his name and say hello to the user.
Do not repeat the greeting in subsequent messages.

When you start the conversation, you will be provided with a {object} answer to the user any question related to health, medicine, exercise, nutrition, and wellness. 
ONLY answer questions related to health, medicine, exercise, nutrition, and wellness. 
If a question is not related to these topics, politely decline to answer and explain that you can only provide information about medical and fitness topics.

if user asked to change a value in the user_data, update it.
if user asked to a provide another exercise in the exercise_plan, provide it to the user based on his BMI,BMI_case,fitness_gaol ,fitness_level and ask him do you want me to save it or not it .
if user asked to show his exercise_plan, show it to the user in details and in a structured format.
When showing the exercise plan, format it as a detailed, structured string and return it in the "message" field of a JSON object.

Use this flow to build a personalized plan based on the following user input:
user input : {text}
style : {style}
use this instructions to format your response:
{format_instructions}


"""
conversation_prompt_template = ChatPromptTemplate.from_template(system_message)
response_parser = PydanticOutputParser(pydantic_object=ChatResponse)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chat API"}

class ChatRequest(BaseModel):
    user_input: str
    user_data: dict

@app.post("/chat")
async def chat(chat: ChatRequest):
    user_input = chat.user_input
    user_data = chat.user_data
    format_instructions = response_parser.get_format_instructions()
    user_messages = conversation_prompt_template.format_messages(style=style,
                                                                text=user_input,
                                                                object=user_data,
                                                                format_instructions=format_instructions)
    response = conversation.run(input=user_messages[0].content)
    print(response)
    cleaned_response = re.sub(r'```(?:json)?\s*', '', response).strip()
    cleaned_response = cleaned_response.rstrip("```")
    
    try:
        response_json = json.loads(cleaned_response)
        parsed_response = ChatResponse.parse_obj(response_json)
        print(parsed_response)
        return parsed_response.message
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)