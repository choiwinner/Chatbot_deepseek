#실행 streamlit run Chatbot_deepseek.py
#실행 streamlit run Chatbot_deepseek.py --server.address=0.0.0.0 --server.port=8501

import streamlit as st
import os

from langchain_ollama import ChatOllama
import ollama

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as bs
from selenium.webdriver.chrome.options import Options
from webdriver_manager.core.os_manager import ChromeType
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup

import time

#langchain, genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory

from langgraph.graph import START,END,StateGraph
from typing_extensions import TypedDict,List
from langchain_core.prompts.chat import ChatPromptTemplate

def get_driver():


    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')

    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,   #like Gecko) Chrome/58.0.3029.110 Safari/537.3')

    # WebDriver 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    return driver

def google_translate(query: str) -> str:
    """구글 번역(한국어->영어 번역)"""

    driver = get_driver()

    #headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    search_url = f"https://translate.google.com/?sl=ko&tl=en&text={query}&op=translate"

    driver.get(search_url)

    time.sleep(3)

    result = driver.find_elements(By.CLASS_NAME, 'ryNqvb')[0].text

    return result

def deepl_translate(query: str) -> str:
    """deepl 번역(한국어->영어 번역)"""

    driver = get_driver()

    #headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    search_url = f"https://www.deepl.com/ko/translator-ppc?utm_source=naver&utm_medium=paid&utm_campaign=KO_Naver_Brand&utm_adgroup=KO_Naver_Brand&utm_term=DEEPL&n_media=27758&n_query=DEEPL&n_rank=1&n_ad_group=grp-a001-04-000000045475249&n_ad=nad-a001-04-000000335393648&n_keyword_id=nkw-a001-04-000006578576419&n_keyword=DEEPL&n_campaign_type=4&n_contract=tct-a001-04-000000001044608&n_ad_group_type=5&NaPm=ct%3Dm7gzgcuw%7Cci%3D0zq0001ZupDBaYysA1mW%7Ctr%3Dbrnd%7Chk%3D45533394f7471da3957334fe15af5fd82d683fdc%7Cnacn%3DaoNiBcQ568tr#ko/en-us/{query}"

    driver.get(search_url)

    time.sleep(3)

    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

    results = soup.find_all('span', class_='--l --r sentence_highlight')

    result = results[1].text

    return result

class State(TypedDict):
    question: str
    answer: str

def transfer_ko2en2(state:State):
    question_co = state['question'] # state['question'] -> question_co 
    if st.session_state.translate_model == 'google_translate': 
        question_en = google_translate(question_co)           #영어로 번역
    elif st.session_state.translate_model == 'deepl_translate':
        question_en = deepl_translate(question_co)            #영어로 번역
    else:
        st.error('Translate Model Error')
        st.stop()
    st.write(question_en)                                     #영어 question 출력
    return {'question': question_en}                          #stat

def think_answer(state:State):

    query = state['question'] #state['question'] -> query
    deepseek = ChatOllama(model='deepseek-r1:14b',temperature=st.session_state['temp_0'])
    response = deepseek.invoke(query) #query -> response
    st.write(response.content)
    return {'answer': response.content}


def think_answer_chunk(state:State):

    query = state['question'] #state['question'] -> query
    deepseek = ChatOllama(model='deepseek-r1:14b',temperature=st.session_state['temp_0'])
    response = ''
    sentence = ''
    for chunk in deepseek.stream(query):
        if chunk.content in ['\n\n', '\n\n\n']:
            st.write(sentence)
            sentence = ''
        else:
            sentence = sentence + chunk.content
        
        response = response + chunk.content

    return {'answer': response}


def think_answer_chunk2(state:State):

    query = state['question'] #state['question'] -> query
    deepseek = ChatOllama(model='deepseek-r1:14b',temperature=st.session_state['temp_0'])
    # stream으로 출력하기(https://wikidocs.net/232694)
    think_message_placeholder = st.empty() # DeltaGenerator 반환
    full_response = '' 
    for chunk in deepseek.stream(query):
            full_response += chunk.content
            think_message_placeholder.markdown(full_response)

    #st.write(full_response)    

    return {'answer': full_response}

def transfer_en2ko(state:State):
    
    answer_en = state['answer'] # state['answer'] -> question_en
    transfer_prompt_en2ko = ChatPromptTemplate([
        (
            "system",
            """당신은 사용자의 Query를 한국어로 번역하는 전문가 AI입니다.
            규칙 1: Query의 답변은 하지 않습니다.
            규칙 2: Query 번역 결과만 출력합니다.
            """
        ),
        (
            "human",
            """
            Query: {question}
            """
        )]) 
    messages = transfer_prompt_en2ko.invoke({'question':answer_en})
    exaone = ChatOllama(model='exaone3.5:7.8b',temperature=0) 
    answer_ko = exaone.invoke(messages)               #한국어로 번역 후 answer_ko
    #print(answer_ko.content)                         #영어 answer_ko 출력
    return {'answer': answer_ko.content}             #state['answer']=answer_ko

def transfer_en2ko_chunk(state:State):
    
    answer_en = state['answer'] # state['answer'] -> question_en
    transfer_prompt_en2ko = ChatPromptTemplate([
        (
            "system",
            """당신은 사용자의 Query를 한국어로 번역하는 전문가 AI입니다.
            규칙 1: Query의 답변은 하지 않습니다.
            규칙 2: Query 번역 결과만 출력합니다.
            """
        ),
        (
            "human",
            """
            Query: {question}
            """
        )]) 
    messages = transfer_prompt_en2ko.invoke({'question':answer_en})
    exaone = ChatOllama(model='exaone3.5:7.8b',temperature=0)
    response = ''
    sentence = ''
    for chunk in exaone.stream(messages):
        if chunk.content in ['\n\n', '\n\n\n']:
            st.write(sentence)
            sentence = ''
        else:
            sentence = sentence + chunk.content

        response = response + chunk.content

    st.write(response)
        
    return {'answer': response}             #state['answer']=answer_ko

def transfer_en2ko_chunk2(state:State):
    
    answer_en = state['answer'] # state['answer'] -> question_en
    transfer_prompt_en2ko = ChatPromptTemplate([
        (
            "system",
            """당신은 사용자의 Query를 한국어로 번역하는 전문가 AI입니다.
            규칙 1: Query의 답변은 하지 않습니다.
            규칙 2: Query 번역 결과만 출력합니다.
            """
        ),
        (
            "human",
            """
            Query: {question}
            """
        )]) 
    messages = transfer_prompt_en2ko.invoke({'question':answer_en})
    exaone = ChatOllama(model='exaone3.5:7.8b',temperature=0)

    transfer_en2ko_message_placeholder = st.empty() # DeltaGenerator 반환
    full_response = '' 
    for chunk in exaone.stream(messages):
            full_response += chunk.content
            transfer_en2ko_message_placeholder.markdown(full_response)

    #st.write(full_response)

    return {'answer': full_response}             #state['answer']=full_response


def langgraph_build():
    builder = StateGraph(State) # 그래프 생성
    
    #step3. node 추가하기
    builder.add_node("transfer_ko2en", transfer_ko2en2)
    builder.add_node("transfer_en2ko", transfer_en2ko_chunk2)
    builder.add_node("think_answer", think_answer_chunk2)
    
    #step4. edge 정의하기
    builder.add_edge(START, "transfer_ko2en")
    builder.add_edge("transfer_ko2en", "think_answer")
    builder.add_edge("think_answer", "transfer_en2ko")
    builder.add_edge("transfer_en2ko",END)

    #step5. graph compile하기
    graph = builder.compile()

    return graph

def main():

    #os.environ['KMP_DUPLICATE_LIB_OK']='True'   # 충돌 제어 주의 : 성능 저하/오류 가능성 있음

    #st.set_page_config(page_title="Lagnchain_with_pdf", page_icon=":books:")

    st.title("💬 Chatbot_deepseek")
    st.caption("🚀 A Streamlit chatbot powered by deepseek")

    #1. st.session_state 초기화
    if "messages_0" not in st.session_state:
        st.session_state['messages_0'] = []
    if "chat_history_0" not in st.session_state:
        st.session_state.chat_history_0 = []
    if "memory_0" not in st.session_state:
        #윈도우 크기 k를 지정하면 최근 k개의 대화만 기억하고 이전 대화는 삭제
        st.session_state.memory_0 = ConversationBufferWindowMemory(memory_key="chat_history",
                                                                   k=4)
    if "temp_0" not in st.session_state:
        st.session_state.temp_0 = 0

    if "translate_model" not in st.session_state:
        st.session_state.translate_model = ''
    
    with st.sidebar:

        if data_clear :=st.button("대화 클리어"):
            st.session_state['messages_0'] = [] #st.session_state[messages]를 초기화
            st.session_state.chat_history_0 = []

        st.session_state['temp_0'] = st.slider("llm 생성시 창의성을 조절합니다. 1에 가까울수록 창의적인 출력이 생성됩니다", 0.0, 1.0, 0.0)

        st.info(f"선택된 llm 창의성은 {st.session_state['temp_0']} 입니다.")

        st.session_state['translate_model'] = st.radio("Translate Model 선택", ('google_translate', 'deepl_translate'), key="model_0")

        st.info(f"선택된 번역 모델은 {st.session_state['translate_model']} 입니다.")


    st.chat_message("assistant").write("안녕하세요. 무엇을 도와드릴까요?")

    graph = langgraph_build()

    #3. query를 입력받는다.
    if query := st.chat_input("질문을 입력해주세요."): 

        #4.'user' icon으로 query를 출력한다.
        st.chat_message("user").write(query)

        #5. query를 session_state 'user'에 append 한다.
        st.session_state['messages_0'].append(('user',query))
        
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                # chain 호출
                start_time = time.time()
                response = graph.invoke({'question':query})
                #st.write(response['answer'])
                end_time = time.time()
                total_time = (end_time - start_time)
                st.info(f"검색 소요 시간: {total_time}초")
                st.session_state['messages_0'].append(('assistant',response['answer']))

if __name__ == '__main__':
    main()