import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# 🔧 Load environment variables
load_dotenv()

# 🧠 Setup tool wrappers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

# 🧰 Named tools
search_tool = DuckDuckGoSearchRun(name="Search")
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper, name="Arxiv")
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper, name="Wikipedia")

tools = [search_tool, arxiv_tool, wiki_tool]

# 🎨 UI
st.title("🔎 LangChain - Chat with Web Tools")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# 🗂️ Initialize chat memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# 🧾 Display message history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 🗣️ New message input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 🧠 Load Groq LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True
    )

    # ⚙️ Agent setup with prompt guidance
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": "You are a helpful agent. Use tools to answer the user query. "
                      "Always give a final answer in plain English after using tools."
        }
    )

    # 💬 Agent response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            response = search_agent.run(prompt, callbacks=[st_cb])
            if not response.strip() or "Complete!" in response:
                raise ValueError("Agent did not return a meaningful answer.")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

        except Exception as e:
            st.warning("🤖 The agent couldn't generate a proper answer. Here's a direct result from DuckDuckGo:")
            fallback = search_tool.run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
            st.write(fallback)
