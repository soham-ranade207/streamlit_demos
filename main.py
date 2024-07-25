import streamlit as st
import openai
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

if "knowledge_graph" not in st.session_state:
    st.session_state["knowledge_graph"] = {}

st.title("Finance Domain Chat Assistant")

def reset_knowledge_graph():
    st.session_state["knowledge_graph"] = {}
    st.experimental_rerun()

def reset_conversation():
    st.session_state["messages"] = [system_message1]
    system_message2["content"] = system_message2["content"].format(
            knowledge_graph=st.session_state["knowledge_graph"]
        )
    st.session_state["messages"].append(system_message2)
    st.session_state["waiting_for_input"] = False
    st.session_state["current_question"] = "What can I help with today?"
    st.session_state["conversation_ended"] = False

api_key = st.sidebar.text_input("Enter your OpenAI API key:")

if api_key:
    client = openai.OpenAI(api_key=api_key)
    
    st.sidebar.title("Knowledge Graph")
    if st.session_state["knowledge_graph"]:
        for key, value in st.session_state["knowledge_graph"].items():
            st.sidebar.text_input(key, value, key=f"kg_{key}")
    else:
        st.sidebar.write("No entries in the knowledge graph yet.")

    if st.sidebar.button("Reset Knowledge Graph"):
        reset_knowledge_graph()

    system_message1 = {
        "role": "system",
        "content": """
You are an AI chat assistant who is an expert in the finance domain, responsible for helping write SQL queries. Your task is to choose from 3 different functions to fill the gaps between the user's question and ensure there's enough information to write a SQL query based on it. You will not be providing the SQL queries themselves.
The 3 functions are:

Ask the user for the first question.
Ask the user a follow-up question with options to disambiguate and better understand the user's query.
Stop processing, which will send the final question response back and also send a knowledge_piece dictionary to be added to the knowledge graph for further interactions.

Your domain is strictly limited to the following tables and their schemas and dimension information:
[Include the existing table schemas and information here]
You will also be provided with a knowledge_graph in a Python dict format with ('key':value) pairs as additional domain knowledge.
Please follow these rules:

Always start by asking the user for a question.
Identify and clarify jargon terms, which are defined as:
a. Words that are not part of the defined domain (e.g., "Budget Variance").
b. Words that are ambiguous in translating into SQL (e.g., "top performing products", "major locations").
c. Words that are interpretable but may be misunderstood (e.g., product major appliances vs. product "major appliances").
If dangling names are provided which don't refer to specific entities in your domain, ask which dimension they belong to.
Do not map similar-sounding or semantically similar categories to valid values. For example, 'Bonus' should not be mapped to 'Personnel Expenses'. Ask the user to help disambiguate and add to the knowledge graph.
Make reasonable assumptions with fiscal years.
Use the provided knowledge graph.
Only ask questions based on the defined scope and the provided knowledge graph.
You can only answer questions based on revenues, expenses, profitability analysis, variance analysis, and sales. Any other finance references should be disambiguated.
Disambiguate based on the dimensions and the knowledge graph provided to you. Clarify with the user when necessary.
Do not assume similar-sounding dimensions are the same (e.g., departments should not be confused with cost centers and profit centers).
When suggesting additions to the knowledge graph, ensure you're not adding the same term twice (e.g., "Major Region" and "major region" are the same).
For ambiguous terms or jargon, provide options or ask for clarification to ensure precise understanding.
If a term is not in the defined schema or knowledge graph, ask the user to clarify or provide more context.
When encountering potentially interpretable but incorrect terms (like "product major appliances"), ask the user if they mean the product category "major appliances" or if it's a specific product name.
For terms that could have multiple interpretations within the finance domain, provide options and ask the user to choose the intended meaning.
Remember, your goal is to gather enough clear and unambiguous information to formulate a precise SQL query, even though you won't be writing the query yourself.
        """,
    }

    system_message2 = {
        "role": "system",
        "content": """
        This is the current knowledge graph to use:
        knowledge_graph:{knowledge_graph}
        """,
    }

    # defining the tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "stop_processing",
                "description": "Answer the user question , add to the knowledge graph and reset the messages queue",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": [""]},
                                    "content": {"type": "string", "enum": [""]},
                                },
                            },
                            "description": 'Messages is a dummy object for function calling. Pass [{"role":"","content":""}]',
                        },
                        "knowledge_pieces": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "jargon": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                            },
                            "description": "These will only be Specifc jargon words that we have disambiguated with user inputs. Only include terms that are uncommon. Should be case insensitive while adding a knowledge pieces. Do not add repeat jargon words in the knowledge graph.",
                        },
                    },
                },
                "required": ["messages", "knowledge_pieces"],
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_for_followup",
                "description": "Function which will ask for a follow up question if the user question is not clear. Provides options for the user to choose from.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": [""]},
                                    "content": {"type": "string", "enum": [""]},
                                },
                            },
                            "description": 'Messages is a dummy object for function calling. Pass [{"role":"","content":""}]',
                        },
                        "assistant_question": {
                            "type": "string",
                            "description": "The follow up question that the LLM will ask to answer user question.",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of options for the user to choose from.",
                        },
                    },
                    "required": ["messages", "assistant_question", "options"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Function which will be used to ask the user to ask a new question. Should be called when the LLM doesnt have an idea about user question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": [""]},
                                    "content": {"type": "string", "enum": [""]},
                                },
                            },
                            "description": 'Messages is a dummy object for function calling. Pass "messages":[{"role":"","content":""}]',
                        }
                    },
                    "required": ["messages"],
                },
            },
        },
    ]

    def stop_processing(messages, knowledge_pieces=[]):
        messages.append(
            {
                "role": "user",
                "content": """
You are an helpful LLM. 
You will be provided with a context and your task is to only return a well defined user question which will summarize the context perfectly if the context is relevant to the scope defined earlier.
Based on the knowledge graph you will also substitue the terms to their relevant meaning. 
You will return a well formed query based on previous context provided to you.
If the context is valid based on the scope follow the following rules:
- Make sure to specify the type of the entity along with the name.
- Return only the final question. You dont have to provide any answers. Your task is to only summarize the context into one final user question.
If you are returning a question use 'Question':....
            """,
            }
        )
        try:
            for knowledge_piece in knowledge_pieces:
                st.session_state["knowledge_graph"][knowledge_piece["jargon"]] = knowledge_piece["value"]
        except Exception as e:
            logging.error(f"Error updating knowledge graph: {e}")
        try:
            response = client.chat.completions.create(model="gpt-4o", messages=messages)
            logging.info(f"Response in stop processing called: {response}")
            final_question = json.dumps(response.choices[0].message.content, indent=2)
            return final_question
        except Exception as e:
            logging.error(f"Error in stop_processing: {e}")
        return "Error occurred while processing the question."
        

    def process_user_input(question, options=None):
        st.write(question)
        if options:
            choice = st.radio(
                "Choose an option or select 'Other' to provide your own input:",
                options + ["Other"],
            )
            if choice == "Other":
                return st.text_input("Please provide your own input:")
            else:
                return choice
        else:
            return st.text_input("Your response:")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [system_message1]
        system_message2["content"] = system_message2["content"].format(
        knowledge_graph=st.session_state["knowledge_graph"]
    )
        st.session_state["messages"].append(system_message2)
    if "waiting_for_input" not in st.session_state:
        st.session_state["waiting_for_input"] = False
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = "What can I help with today?"
    if "conversation_ended" not in st.session_state:
        st.session_state["conversation_ended"] = False
    if "follow_up_options" not in st.session_state:
        st.session_state["follow_up_options"] = None

    st.write("Chat History:")
    for message in st.session_state["messages"][1:]:  # Skip the system message
        st.write(f"{message['role'].capitalize()}: {message['content']}")

    if st.session_state["conversation_ended"]:

        if st.button("Start New Conversation"):
            reset_conversation()
            st.experimental_rerun()

    elif st.session_state["waiting_for_input"]:
        user_input = process_user_input(
            st.session_state["current_question"], st.session_state["follow_up_options"]
        )
        if st.button("Submit"):
            st.session_state["messages"].append(
                {"role": "assistant", "content": st.session_state["current_question"]}
            )
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["waiting_for_input"] = False
            st.session_state["follow_up_options"] = None  # Reset options after use
            st.experimental_rerun()
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state["messages"],
                tools=tools,
                tool_choice="required",
            )
            response_message = response.choices[0].message
            if response_message.tool_calls:
                function_name = response_message.tool_calls[0].function.name
                function_params = json.loads(
                    response_message.tool_calls[0].function.arguments
                )

                logging.info(f"Function called: {function_name}")
                logging.info(f"Function parameters: {function_params}")

                if function_name == "stop_processing":
                    final_question = stop_processing(
                        st.session_state["messages"], function_params.get("knowledge_pieces",[])
                    )
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": f"{final_question}"}
                    )
                    st.session_state["conversation_ended"] = True
                elif function_name == "ask_for_followup":
                    st.session_state["current_question"] = function_params.get(
                        "assistant_question", "What can I help with today?"
                    )
                    st.session_state["follow_up_options"] = function_params.get(
                        "options"
                    )
                    st.session_state["waiting_for_input"] = True
                elif function_name == "ask_user":
                    st.session_state["current_question"] = "What can I help with today?"
                    st.session_state["waiting_for_input"] = True
                st.experimental_rerun()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            st.error("An error occurred. Please try again.")
