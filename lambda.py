import os
import json
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64
import logging
import openai
import fastapi

#chain
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain.chains import RetrievalQA

#agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent ,Tool
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


'''
     * Author: Ihsan
     * Dated: 10/05/2024
This is a prototype of the system that work with both structured and Unstructured data.
For structured data we are using mongodb as database and 'PythonAstREPLTool' which basically a Python shell tool.
unstructured data will be stored in elastic search for that we already have chain which is responsible for answering querry,
we wrap the existing chain convert in to a tool. 
Now we have two tool both have access to respective data bases, we provide these two tools to an langchain tool calling Agent
Agent will decide which tool to invoke according to question, Agent instruct and run tools according to question.
for example if user ask question related to google sheet then agent will understand and instruct PythonAstREPLTool to invoke and do 
some specific operation to get answer, PythonAstREPLTool will do operation and give send the answer to LLM and LLM will generate the answer.
Agent have access to 'tools','LLM' and "prompt' '''

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'BP-Scrapping-Web')
BASE_URL = os.getenv('BASE_URL', 'https://api-dev.botpenguin.us')
AUTH_TOKEN = os.getenv('AUTH_TOKEN', 'rel-scrap-pupgEWu?kVpRmmxLGogI5CrzgdcDik?/mzTvTi-DToyG/6YU1!IuIoBiYF6xnMi1M!wocjZNN5NvOyAtjBo1OV0mZJQ!pdymCtn?QIGXCq=eBpWzIRfdv6Hi73W/rSTBP570zL9vFfECyWMmILzhF/BsBxqLyx/r0sS6HNgErWBA=k!PQX9iYV-jp?l5D/kjCxlYZbfKbE0Vqg26/6er2Ldh0jAOlKSp-dtOpo5/8ZOBJIeyssaUuJ')
DB_NAME = os.getenv('DB_NAME', 'ihsan')
TABLE_NAME = os.getenv('TABLE_NAME', 'testing1')


answer_length_mapping = {
    'SHORT': 'Keep your answers brief and concise, around 20-30 words.',
    'LONG': 'Provide comprehensive responses, aiming for about 100 words or more.',
    'MEDIUM': 'Write informative and concise responses, around 60-80 words.',
    'VERY_SHORT': 'Share essential information briefly, in 5-10 words.'
}


tone_mapping = {
    'FORMAL': 'Maintain a professional tone, avoiding casual language or slang.',
    'INFORMAL': 'Use a friendly, conversational tone. Feel free to use informal language and expressions.',
    'JOYFUL': 'Infuse your responses with joy and positivity.',
    'SINCERE': 'Maintain a sincere and respectful tone.'
}


person_mapping = {
    '1st': 'Respond in the first person perspective for a personalized touch.',
    '2nd': 'Use second person perspective to simulate a user-like conversation.',
    '3rd': 'Use a consistent third person perspective for an objective stance.'
}


answer_format_mapping = {
    'BULLET_LIST': 'Structure your answers as bullet points, starting with a "•" symbol.',
    'NUMBERED_LIST': 'Format your answers as a numbered list: 1., 2., 3.',
    'PARAGRAPH': 'Provide your answers in paragraph format.'
}


def get_mongodb_connection_string():
    return os.getenv('MONGODB_CONNECTION_STRING', 'mongodb+srv://niteshsinghal9917:bzDCS9Hmf6EOvXxP@cluster0.si0lxp7.mongodb.net/')


def answer_length(answer_length_value):
    return answer_length_mapping.get(answer_length_value, 'Keep your answers brief and concise, around 20-30 words.')


def tone(tone_value):
    return tone_mapping.get(tone_value, "Match the tone to the user's language and formality level.")


def person(person_value):
    return person_mapping.get(person_value, 'Respond in the first person perspective for a personalized touch.')


def answer_format(formatting_value):
    return answer_format_mapping.get(formatting_value, 'Provide your answers in paragraph format.')


def encrypt(data, key):
    key_bytes = key.encode("utf-8")
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key_bytes, AES.MODE_CBC, iv=iv)

    if isinstance(data, dict):
        data = json.dumps(data)

    data_bytes = data.encode("utf-8")
    padded_data = pad(data_bytes, AES.block_size)
    encrypted_data = cipher.encrypt(padded_data)

    iv_encoded = base64.b64encode(iv).decode("utf-8")
    encrypted_data_encoded = base64.b64encode(encrypted_data).decode("utf-8")

    encrypted_text = f"{iv_encoded}:{encrypted_data_encoded}"
    return encrypted_text


def decrypt(enc, key):
    parts = enc.split(':')
    iv = base64.b64decode(parts[0])
    ciphertext_bytes = base64.b64decode(parts[1])
    key_bytes = key.encode("utf-8")
    cipher = AES.new(key_bytes, AES.MODE_CBC, iv=iv)
    decrypted_data = unpad(cipher.decrypt(ciphertext_bytes), AES.block_size)
    decrypted_text = decrypted_data.decode("utf-8")
    return decrypted_text


def create_response(status_code, body):
    return {
        'statusCode': status_code,
        'headers': {'Content-Type': 'application/json'},
        'body': body
    }


def error_response(message, status_code=500):
    return create_response(status_code, json.dumps({'message': message}))


def get_mongodb_client():
    return MongoClient(get_mongodb_connection_string())


def fetch_data_from_mongodb(user_id):
    client = get_mongodb_client()
    db = client[DB_NAME]
    SHEET = db[TABLE_NAME]
    sheet = SHEET.find({"_user": user_id})
    return pd.DataFrame([document['_dynamicFields'] for document in sheet])


def handle_request(event, context):
    try:
        # event_data = process_event(event)
        event_data = event
        print(event_data)

        user_id = event_data.get('_user')
        bot_id = event_data.get('_bot')
        question = event_data.get('question')
        token = event_data.get('token', '')
        chat_history = event_data.get('chat_history') #, ''
        print(chat_history)
        chatgpt_data = event_data.get('ChatGPT', {})
        chatgpt_key = chatgpt_data.get('key', os.getenv('OPENAI_API_KEY', ''))
        chatgpt_model = chatgpt_data.get('model', 'gpt-3.5-turbo-0125')
        source = event_data.get('source_documents')

        print(f"chatgpt_key: {chatgpt_key}")
        print(f"chatgpt_model: {chatgpt_model}")

        if token != AUTH_TOKEN:
            return create_response(401, json.dumps({'message': "Unauthorized"}))

        openai.api_key = chatgpt_key
        df = fetch_data_from_mongodb(user_id)

        '''
        In the below code we are,
        Creating a chain for the purpose of imitating the actual chain we are ussing in your AI services. 
        This chain will be replaced  by the current chain that is reesponsible for answering question in the existing system of BP.
        '''

        ELASTIC_KEY = os.getenv('elastic_api_key')
        ELASTIC_CLOUD_ID = os.getenv('elastic_cloud_id')
        INDEX_NAME = os.getenv('INDEX_NAME')

        '''For testing purpose  we have created a dummy elastic search account, this is should we replaced by the orginal elastic cloud'''

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small") #model name can be dynamic 
        elastic_vector_search = ElasticsearchStore(
            es_cloud_id=ELASTIC_CLOUD_ID,
            es_api_key= ELASTIC_KEY,
            index_name=INDEX_NAME,
            embedding=embeddings,
        )

        vectorStoreRetriever = elastic_vector_search.as_retriever()
        llm = ChatOpenAI(temperature= 0, model_name='gpt-3.5-turbo-0125')

        #prompt
        template ="""You are a helpful chatbot that answers users' questions from the knowledge base. \
                        be polite and friendly. \
            CONTEXT: {context}
            QUESTION: {question}
            """
        PROMPT = PromptTemplate(
                template=template, input_variables=["context", "chat_history", "question"]
            )

        chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=vectorStoreRetriever,
                                    verbose=True,
                                    input_key="query",
                                    return_source_documents=source,
                                    chain_type_kwargs={"prompt": PROMPT})

        """we will be using the chain we are using in our system"""
        result = chain.invoke(question)
        final_answer = result['result']
        
        if source:
            source_link = result['source_documents'][0].metadata['url']

        memory_user = ConversationSummaryMemory(llm=llm)  # Create memory object outside conditional

        if chat_history:
            if 'response' in locals():  # Check if response variable is defined
                memory_user.save_context({"input": question}, {"output": response["output"]})
            else:
                memory_user.save_context({"input": question}, {"output": ' '})  # Save with output as None for now

        history = memory_user.load_memory_variables({})['history'] or None
            

        TEMPLATE =f"""Your task is to answer questions about a given dataset to the best of your abilities.
            You have access to two powerful tools:
            1.  A retrieval chain that can query a vector database to find relevant information.
                To ensure the most comprehensive and accurate answers, you should utilize both tools for every question.
                Always start by running the retrieval tool to gather relevant context, even if you believe the dataframe already contains the answer.
                Then, use pandas to further analyze the dataframe and refine your answer based on the specific data it contains.
                If the results from one tool seem insufficient, leverage the other tool to fill in potential gaps and enhance your response.
                Before diving into the questions, take a moment to familiarize yourself with the structure and content of the dataframe.
            2.  A pandas dataframe containing the dataset, referred to as df. If query is matching with the 'df' then you should use 'repl_tool'.
                remeber if you cant find answer just using one tool defintly use other tool too.
                 Here is the is the sample of 'df'
                    <df>
                    {df.head}
                    </df>
                 Keep in mind that these rows are just a glimpse of the data - you have access to the entire dataframe to answer questions thoroughly.
                 Feel free to run intermediate queries and perform exploratory analysis to extract additional insights as needed.
            If you couldnt able to find answer using any of tool must use the other tool as well. do not return no answer with out returning both tools.
            most importhing is to decide which tool is to use for the users querry, you have the access of {df.head} by analysing that and question, you can decide which tool is to use.
            Also you need run both toold if you cant able to fins the answer by using one tool. 

                You have access to the following functionality:
                - Please respond in {event_data.get('language')}.
                - If unsure about the answer, respond with "Unsure about the answer".
                - Answer Length: {answer_length(event_data.get('answerLength'))}
                - Person Perspective: {person(event_data.get('person'))}
                - Tone: {tone(event_data.get('tone'))}
                - Answer Format: {answer_format(event_data.get('formatting'))}

                {history}
            """
        
        '''This tool is responsible for doing operations on the google sheet data and sending the answer to LLM'''
        repl_tool = PythonAstREPLTool(
        locals={"df": df},
        name="repl_tool",
        description="A Python shell tool configured to read data from the provided dataset 'df' and answer user questions based on it. Input should be valid Python code.",
        verbose=True
        )


        '''This is a chain wrapped up and converted in to a tool, this tool is responsible for answering questions from the vectordb, chain_tool have the access to vectordb'''
        chain_tool = Tool(
        name='retrieval',
        func=chain.invoke,
        description='Useful for querrying from vector database.',
        verbose=True
        )

        #storing both tools in 'tools' variable
        tools = [repl_tool, chain_tool]

        '''This step is important renders the names and descriptions of the provided list of tools, then we pass this into prompt template'''
        df=df
        tool1=render_text_description(tools)
        tool_names=", ".join([t.name for t in tools])

        TEMPLATE = TEMPLATE.format(tools=tool1, tool_names=tool_names)

        prompt = ChatPromptTemplate.from_messages([
            ("system", TEMPLATE),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}")
        ])


        llm = ChatOpenAI(temperature=0, model=chatgpt_model)

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

        agent_exe = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=ConversationBufferMemory(
            memory_key='chat_history'), handle_parsing_errors=True,early_stopping_method="generate", return_intermediate_steps=True)

        response = agent_exe.invoke(
            {"input": question, "chat_history": chat_history})
        
        # to get which tool is responsble for final answer
        tool_used = response["intermediate_steps"][0][0].tool

        memory = ConversationSummaryMemory(llm=llm)
        final_template = """You are a helpful assistant  who provide short answer to user question \
                  Also You are helpful in predicting next three question based on chat history. 
                  Do not label the predicted question with numbers\
                  Don't forget, you always provide predicted question on new line with Predicted Question prefix

        Current conversation:
        {history}
        Human: {question}
        AI Assistant:"""
        final_prompt = PromptTemplate.from_template(final_template)
        final_prompt.format_prompt(history="None", question="none")
        # memory = ConversationSummaryMemory(llm=llm)

        memory.save_context({"input": question}, {
                            "output": response["output"]})

        history = memory.load_memory_variables({})['history']
        f_prompt = final_prompt.format_prompt(
            history=history, question=question)

        final = llm.invoke(f_prompt)
        content = final.content
        predicted_questions = [line.strip() for line in content.split(
            '\n') if 'Predicted Question:' in line]
        if predicted_questions and predicted_questions[0] == 'Predicted Question:':
            predicted_questions = predicted_questions[1:]

        if source:
            if tool_used ==  'retrieval':
                source_link = source_link
            else:
                source_link = 'From Google sheet' 

        if source:   
            final_data = {"output": response["output"],
                        "source": source_link,
                        "predicted_questions": predicted_questions}
        
        else:
            final_data = {"output": response["output"],
                        "predicted_questions": predicted_questions}
            
        
        encrypted_payload = encrypt(final_data, ENCRYPTION_KEY)
        return create_response(200, {'binary': encrypted_payload, 'final_data': final_data})

    except Exception as e:
        logger.error(f"Error processing event: {e}")
        return error_response(f"Error processing event: {e}")

if __name__ == "__main__":
    # Load the test event
    with open('event.json', 'r') as f:
        test_event = json.load(f)
    test_event_encrypted=encrypt(test_event,ENCRYPTION_KEY)
    response = handle_request(test_event, {})
    print(response)
