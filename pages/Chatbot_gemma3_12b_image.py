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
from langchain_core.messages import HumanMessage


from PIL import Image
import base64
import io
from io import BytesIO

class State(TypedDict):
    question: str
    answer: str
    encoded_image : str
    image_format : str

def think_answer_chunk2(state:State):

    query = state['question']
    encoded_image = state['encoded_image']
    format = state['image_format']

    llm = ChatOllama(model=st.session_state.llm_model,
                     temperature=st.session_state['temp_0'])
    
    chain = prompt_func | llm

    # stream으로 출력하기(https://wikidocs.net/232694)
    think_message_placeholder = st.empty() # DeltaGenerator 반환
    full_response = '' 
    for chunk in chain.stream({'question': query,
                            'image': encoded_image,
                            'image_format': format}):
            
            full_response += chunk.content
            think_message_placeholder.markdown(full_response)

    return {'answer': full_response}

def image_upload():
    
    # 파일 업로더 위젯
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    # 파일이 업로드되면 처리
    if uploaded_file is not None:

        # 파일을 바이너리 데이터로 읽기
        img_bytes = uploaded_file.read()

        # PIL로 이미지를 열고 스트림릿 앱에 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.")
        st.write("Image uploaded successfully!")
    else:
        st.write("Please upload a JPG or PNG file.")

    buffered = BytesIO()
    format = image.format.lower()  # Get the file format (e.g. "jpg", "jpeg", "png")
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str,format

def prompt_func(data):
    text = data['question']
    image = data['image']
    format = data['image_format']

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/{format};base64,{image}",
    }

    content_parts = []

    text_part = {
        "type": "text",
        "text": text
    }

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

def langgraph_build():
    builder = StateGraph(State) # 그래프 생성
    
    #step3. node 추가하기
    builder.add_node("think_answer", think_answer_chunk2)
    
    #step4. edge 정의하기
    builder.add_edge(START, "think_answer")
    builder.add_edge("think_answer",END)

    #step5. graph compile하기
    graph = builder.compile()

    return graph

def main():

    #os.environ['KMP_DUPLICATE_LIB_OK']='True'   # 충돌 제어 주의 : 성능 저하/오류 가능성 있음

    #st.set_page_config(page_title="Lagnchain_with_pdf", page_icon=":books:")

    st.title("💬 Chatbot_exaone-deep")
    st.caption("🚀 A Streamlit chatbot powered by exaone-deep")

    #1. st.session_state 초기화
    if "messages_0" not in st.session_state:
        st.session_state['messages_0'] = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory_0" not in st.session_state:
        #윈도우 크기 k를 지정하면 최근 k개의 대화만 기억하고 이전 대화는 삭제
        st.session_state.memory_0 = ConversationBufferWindowMemory(memory_key="chat_history",k=4)
    if "temp_0" not in st.session_state:
        st.session_state.temp_0 = 0

    if "translate_model" not in st.session_state:
        st.session_state.translate_model = ''

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = ''

    if "messages_img" not in st.session_state:
        st.session_state['messages_img'] = [] 
        #st.session_state에 messages가 없으면 빈 리스트로 초기화

    if "chat_img" not in st.session_state:
        st.session_state.chat_img = ''

    if "image_format" not in st.session_state:
        st.session_state.image_format = ''
        
    with st.sidebar:

        model_name = []

        # 모델 목록 가져오기
        models = ollama.list()

        # 모델 목록 출력하기
        for model in models['models']:
            #st.write(model['model'])
            model_name.append(model['model'])

        st.session_state.llm_model = st.sidebar.selectbox('Choose LLM Model', 
                                                        model_name,key='selected_model')
        
        st.info(f'{st.session_state.llm_model}을 선택하셨습니다.')

        if data_clear :=st.button("대화 클리어"):
            st.session_state['messages_0'] = [] #st.session_state[messages]를 초기화
            st.session_state.chat_history = []
            st.session_state['messages_img'] = []
            st.session_state.chat_img = ''
            st.session_state.image_format = ''

        st.session_state['temp_0'] = st.slider("llm 생성시 창의성을 조절합니다. 1에 가까울수록 창의적인 출력이 생성됩니다", 0.0, 1.0, 0.0)

        st.info(f"선택된 llm 창의성은 {st.session_state['temp_0']} 입니다.")

        st.session_state['translate_model'] = st.radio("Translate Model 선택", ('google_translate', 'deepl_translate'), key="model_0")

        st.info(f"선택된 번역 모델은 {st.session_state['translate_model']} 입니다.")

    #2. graph build
    graph = langgraph_build()

    # 이미지 로드
    st.session_state.chat_img, st.session_state.image_format = image_upload()

    st.chat_message("assistant").write("안녕하세요. 무엇을 도와드릴까요?")

    #2. 이전 대화 내용을 출력
    # st.session_state['chat_history']가 있으면 실행
    if ("chat_history" in st.session_state) and (len(st.session_state['chat_history'])>0):
        #st.session_state['messages']는 tuple 형태로 저장되어 있음.
        for role, message in st.session_state['chat_history']: 
            st.chat_message(role).write(message)

    #3. query를 입력받는다.
    if query := st.chat_input("질문을 입력해주세요."): 

        #4.'user' icon으로 query를 출력한다.
        st.chat_message("user").write(query)

        #5. query를 session_state 'user'에 append 한다.
        st.session_state['chat_history'].append(('user',query))

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                # chain 호출
                start_time = time.time()
                response = graph.invoke({'question':query,
                                         'encoded_image':st.session_state.chat_img,
                                         'image_format':st.session_state.image_format})

                end_time = time.time()
                total_time = (end_time - start_time)
                st.info(f"검색 소요 시간: {total_time}초")
                st.session_state['chat_history'].append(('assistant',response['answer']))

if __name__ == '__main__':
    main()