import os
import gc # Garbage Collector - TO GET GIT OF OBJECTS THAT ARE NO LONGER NEEDED.
import time
import base64
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool # For Web-Search.

from tools.custom_tools import DocumentSearchTool

@st.cache_resource
def load_llm():
    llm = LLM(
        model="ollama/llama3.2",
        base_url="http://localhost:11434"
    )
    
    return llm

# DEFINE AGENTS AND TASKS.
def create_agents_and_tasks(pdf_tool):
    web_search_tool = SerperDevTool()
    retriever_agent = Agent(
        role = 'Retrieve relevant information to answer the user query: {query} ',
        goal = (
            'Retrieve the most relevant information from the available sources '
            'for the user query: {query}. Always try to use the PDF search tool first. '
            'If you are not able to retrieve the information from the PDF search tool, '
            'then try to use the web search tool.'
        ),
        backstory = (
            'You are a meticulous analyst with a keen eye for detail.'
            'You are known for your ability to understand user queries: {query}'
            'and retrieve knowledge from the most suitable knowledge base.'
        ),
        verbose= True,
        tools = [tool for tool in [pdf_tool, web_search_tool] if tool],
        llm = load_llm()
    )
    
    response_synthesizer_agent = Agent(
        role = 'Response synthesizer agent for the user query: {query}',
        goal = (
            'Synthesize the retrieved information into a concise and coherent response '
            'based on the user query: {query}. If you are not able to retrieve the '
            'information then respond with I \'m sorry, I couldn\'t find the information you\'re looking for.'
        ),
        backstory = (
            "You're a skilled communicator with a knack for turning "
            "complex information into clear and concise responses."
        ),
        verbose = True,
        llm = load_llm()
    )
    
    retrieval_task = Task(
        description = ('Retrieve the most relevant information from the available sources for the user query: {query}'),
        expected_output = ('The most relevant information in the form of text as retrieved from the sources.'),
        agent = retriever_agent
    )
    
    response_task = Task(
        description = '',
        expected_output = (
            'A concise and coherent response based on the retrieved information from the right source for the user query: {query}.'
            'If you are not able to retrieve the information, then respond with: '
            'I\'m sorry, I couldn\'t find the information you\'re looking for.'
        ),
        agent = response_synthesizer_agent
    )
    
    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew

# STREAMLIT SETUP - USER INTERFACE.

# CHAT HISTORY.
if 'messages' not in st.session_state:
    st.session_state.messages = [] 

# STORE DOCUMENT-SEARCH-TOOL.
if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None 

# STORE CREW OBJECT.
if "crew" not in st.session_state:
    st.session_state.crew = None 
    
def reset_chat():
    st.session_state.messages = []
    gc.collect() # RELEASE MEMORY.

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the Uploaded PDF in an Iframe."""

    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

# SIDE-BAR.
with st.sidebar:
    st.header("Add Your PDF Document")
    pdf_file = st.file_uploader("Choose a PDF file", type = ["pdf"])
    
    if pdf_file is not None: # IF USER UPLODED FILE.
        # IF PDF TOOL IS NOT CREATED YET (FIRST TIME).
        if st.session_state.pdf_tool is None:
            with st.spinner("Processing PDF... Please wait..."):
                st.session_state.pdf_tool = DocumentSearchTool(pdf_file)
    
            st.success("PDF Processed! Ready to chat.")
            
        # OPTIONALLY DISPLAY THE PDF IN THE SIDEBAR.
        display_pdf(pdf_file.getvalue(), pdf_file.name)
        
    st.button('Clear Chat', on_click = reset_chat)
    
    
# MAIN CHAT INTERFACE.

# MARKDOWN SECTION - TO DISPLAY CUSTOM HTML.
st.markdown("""
    # Agentic RAG powered by <img src="data:image/png;base64,{}" width="120" style="vertical-align: -3px;">
""".format(base64.b64encode(open("assets/deep-seek.png", "rb").read()).decode()), unsafe_allow_html=True)
# CHAT INPUT

# RENDER EXISTING CONVERSATION.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
query = st.chat_input('')

if query:
    st.session_state.messages.append({"role": "user", "content": query}) # ADD USER QUERY.
    # DISPLAY USER QUERY.
    with st.chat_message("user"):
        st.markdown(query)
    
    # GET CREW, AFTER PDF IS LOADED.
    if st.session_state.crew is None:
        st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

    # GET THE RESPONSE
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # GET THE COMPLETE RESPONSE FIRST.
        with st.spinner("Thinking..."):
            inputs = {"query": query}
            result = st.session_state.crew.kickoff(inputs=inputs).raw
            
        lines = result.split('\n') # SPLIT THE RESULT BY LINES.
        for i, line in enumerate(lines):
            full_response += line
            
            if i < len(lines) - 1:  # DON'T ADD NEW LINE TO THE LAST LINE.
                full_response += '\n'
                
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.15)
            
        # SHOW THE FINAL RESPONSE WITHOUT CURCOR.
        message_placeholder.markdown(full_response)
        
        # SAVE ASSISTENTS MESSAGE TO SESSION.
        st.session_state.messages.append({"role": "assistant", "content": result})
            
            
            
            
    

    