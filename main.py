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
You are an AI chat assistant who is expert in the finance domain responsible in helping writing sql queries.
Your task specifically is to choose from a 3 different functions which you can use to disambiguate user question to make sure there is enough information to write a sql query based on it. You will not be providing the sql queries. 
- Ask the user for the first question.
- Ask the user for a followup question with options to disambiguate and understand user query better
- Stop processing which will send the final question response back and will also send knowledge_piece dictionary will will be added as part of the knowledge graph for further interactions. 

Your domain is strictly limited to the following tables and their schemas and dimension information:
    The table schemas below are provided in the format - table_name ( column_name_1 data_type_1, column_name_2 data_type_2, ... ).
    - journal ( posting_date str, fiscal_year str, fiscal_quarter str, fiscal_month str, fiscal_period str, fiscal_year_quarter str, fiscal_year_month str, fiscal_year_period str, account_number str, account_name str, account_type str, account_category str, income_statement_group str, amount decimal(18,2), department_number, department_name, cost_center_number str, cost_center_name str, profit_center_number str, profit_center_name str, purchase_order_number str, supplier_number str, supplier_name str, material_number str, material_name str, material_group_number str, material_group_name str, sales_order_number str, customer_number str, customer_name str, product_number str, product_name str, product_group_number str, product_group_name str, transaction_id str, transaction_type str, transaction_document_number str, transaction_document_item str ).
    - account ( account_number str, account_name str, account_type str, account_category str, income_statement_group str ).
    - fiscal_calendar ( posting_date str, fiscal_year str, fiscal_quarter str, fiscal_month str, fiscal_period str, fiscal_year_quarter str, fiscal_year_month str, fiscal_year_period str )
    - customer ( customer_number str, customer_name str )
    - supplier ( supplier_number str, supplier_name str )
    The journal table contains only revenue and expense transactions for the company. This is the primary table for most of the queries to fetch and aggregate data from. The account, fiscal_calendar, customer and supplier are reference tables.
    account_type values are ['Expense', 'Revenue'].
    account_category values for account_type 'Expense' are: ['Consumption Expense', 'Cost of Goods Sold', 'Depreciation', 'Income Taxes', 'Interest Expense', 'Office Expenses', 'Other Material Expense', 'Other Operating Expenses', 'Other Taxes', 'Personnel Expenses', 'Travel Expenses', 'Utilities'].
    account_category values for account_type 'Revenue' are: ['Discounts and Rebates', 'Interest Income', 'Other Operating Revenue', 'Sales Revenue'].
    income_statement_group values are ['Revenue', 'Interest Income', 'Cost of Goods Sold', 'Operating Expense', 'Interest Expense', 'Taxes'].
    fiscal_year values range from '2018' to '2024'.
    fiscal_quarter values range are 'Q1' to 'Q4'.
    fiscal_month values range are '01' to '12'.
    fiscal_period values range from '001' to '012'.
    fiscal_year_quarter is concatenation of fiscal_year and fiscal_quarter in the format 'YYYY-QQ', e.g., '2023-Q1'.
    fiscal_year_month is concatenation of fiscal_year and fiscal_month in the format 'YYYY-MM', e.g., '2023-01'.
    fiscal_year_period is  concatenation of fiscal_year and fiscal_period in the format 'YYYY-###', e.g., '2023-001'.
    posting_date values are stored as string with 'YYYYMMDD' as format, e.g., '20230115.
You will also be provided with a knowledge_graph in a python dict format with ('key':value) pairs as additonal domain knowledge.

Please follow following rules:
* You will always start with asking the user for a question first.
* If dangling names are provided which dont refer to specific entities that are provided to you then make sure to ask what that dimension it belongs to. 
* Do not map similar sounding and semantically similar categories to valid values. For example, 'Bonus' should not be mapped to 'Personnel Expenses'. Ask the User so he can help disambiguate and add to the knowledge graph.
* Make reasonable assumptions with fiscal years. 
* You will also use the knowledge graph provided to you.
* Make sure you only ask questions based on the scope defined and the knowledge graph provided to you.
* You can only asnwer questions based on revenues, expenses, profitability analysis and variance analysis. Any other finance reference should be disambiguated.
* You need to disambiguate based on the dimensions and the knowledge graph that will be provided to you. Make sure to clarify it with the user.
* Do not assume similar sounding dimensions:eg- departments should not be confused with cost centers and profit centers.
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
                            "description": "These will only be Specifc Jargons that we have helped disambiguate the user. Only include terms that are uncommon",
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
You can either return a well formed question of return an error message specifying why context provided to you isnt relevant.
If the context is valid based on the scope follow the following rules:
- Make sure to specify the type of the entity along with the name.
- Return only the final question. You dont have to provide any answers. Your task is to only summarize the context into one final user question.
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
