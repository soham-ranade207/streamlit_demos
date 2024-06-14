import streamlit as st
import openai
import json
import numpy as np
import time
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    handlers=[logging.StreamHandler()],  # Log messages to the console as well
)

api_key = st.text_input("What is your openai api key to use")
if api_key:
    client = openai.OpenAI(api_key=api_key)
    # Define the initial system message
    system_message = {
        "role": "system",
        "content": """
        You are an AI chat assistant who is expert in the finance domain. 
        Your task is to choose from a 3 different functions which you can use to disambiguate user question and understand precisely what the user is asking.
        You have these functions at hand:
        - Ask the user for a new question that you can then answer.
        - Ask the user for a followup question.
        - Stop processing which will clear the messages queue. 

        Your domain is limited to the following tables and their schema:
        The names of tables allowed for SQL generation are [ journal, account, fiscal_calendar, customer, supplier ].
        The table schemas below are provided in the format - table_name ( column_name_1 data_type_1, column_name_2 data_type_2, ... ).
        - journal ( posting_date str, fiscal_year str, fiscal_quarter str, fiscal_month str, fiscal_period str, fiscal_year_quarter str, fiscal_year_month str, fiscal_year_period str, account_number str, account_name str, account_type str, account_category str, amount decimal(18,2), cost_center_number str, cost_center_name str, profit_center_number str, profit_center_name str, department_number, department_name, purchase_order_number str, supplier_number str, supplier_name str, material_number str, material_name str, material_group_number str, material_group_name str, sales_order_number str, customer_number str, customer_name str, product_number str, product_name str, product_group_number str, product_group_name str, transaction_id str, transaction_type str, document_number str, document_item str ).
        - account ( account_number str, account_name str, account_type str, account_category str ).
        - fiscal_calendar ( posting_date str, fiscal_year str, fiscal_quarter str, fiscal_month str, fiscal_period str, fiscal_year_quarter str, fiscal_year_month str, fiscal_year_period str )
        - customer ( customer_number str, customer_name str )
        - supplier ( supplier_number str, supplier_name str )
        The journal table contains only revenue and expense transactions for the company. This is the primary table for most of the queries to fetch and aggregate data from. The account, fiscal_calendar, customer and supplier are reference tables.
        account_type values are [ 'Revenue', 'Expense' ]:
        account_category values for account_type 'Revenue' are: [ 'Change in Inventory', 'Discounts and Rebates', 'Gains Price Difference', 'Other Operating Revenue', 'Sales Revenue' ].
        account_category values for account_type 'Expense' are: [ 'Consumption', 'Cost of Goods Sold', 'Depreciation', 'Interest Expense', 'Office Expenses', 'Other Material Expense', 'Other Operating Expenses', 'Personnel Expenses', 'Travel Expenses', 'Utilities' ].
        fiscal_year values range from '2018' to '2024'.
        fiscal_quarter values range are 'Q1' to 'Q4'.
        fiscal_month values range are '01' to '12'.
        fiscal_period values range from '001' to '012'.
        fiscal_year_quarter is concatenation of fiscal_year and fiscal_quarter in the format 'YYYY-QQ', e.g., '2023-Q1'.
        fiscal_year_month is concatenation of fiscal_year and fiscal_month in the format 'YYYY-MM', e.g., '2023-01'.
        fiscal_year_period is  concatenation of fiscal_year and fiscal_period in the format 'YYYY-###', e.g., '2023-001'.
        posting_date values are stored as string with 'YYYYMMDD' as format, e.g., '20230115.

        Please follow following rules:
        * You will always start with asking the user for a question first.
        * If names are provided then make sure to ask what that entity it belongs to.
        * Make reasonable assumptions with fiscal years.
        * Make sure you only ask questions based on the scope defined. 
        """,
    }

    # defining the tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "stop_processing",
                "description": "Answer the user question and reset the messages queue",
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
                        }
                    },
                    "required": ["messages"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_for_followup",
                "description": "Function which will basically ask for a follow up question if the user question is not clear. There should be a user question before asking for a followup.",
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
                            "description": "The follow up question that the LLM will ask to to answer user question.",
                        },
                    },
                    "required": ["messages", "assistant_question"],
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

    # Define functions for interaction

    def stop_processing(messages):
        messages.append(
            {
                "role": "user",
                "content": """
        You are a helpful LLM.
        You will be provided with a context and your task is to return a well-defined user question which will summarize the context perfectly.
        - Please return just the user question. You don't need to answer the question.
        - Your only task is to form a question which captures the context provided to you.
        - Make sure to specify the type of the entity along with the name.
        """,
            }
        )
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        final_question = json.dumps(response.choices[0].message.content, indent=2)
        messages = []
        return final_question, messages

    def ask_for_followup(messages, assistant_question):
        user_input = st.text_input(
            assistant_question, key=np.random.randint(low=100001, high=200000, size=1)
        )
        if st.button("Submit"):
            messages.append({"role": "assistant", "content": assistant_question})
            messages.append({"role": "user", "content": user_input})
            return messages

    def ask_user(messages):
        user_input = st.text_input(
            "What can I help with today?",
            key=np.random.randint(low=1, high=100000, size=1),
        )
        if st.button("Submit"):
            messages.append(
                {"role": "assistant", "content": "What can I help with today?"}
            )
            messages.append({"role": "user", "content": user_input})
            return messages

    # Streamlit app layout
    st.title("Finance Domain Chat Assistant")

    # User input for the question

    while True:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "stop_processing",
                    "description": "Answer the user question and reset the messages queue",
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
                            }
                        },
                        "required": ["messages"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_for_followup",
                    "description": "Function which will basically ask for a follow up question if the user question is not clear. There should be a user question before asking for a followup.",
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
                                "description": "The follow up question that the LLM will ask to to answer user question.",
                            },
                        },
                        "required": ["messages", "assistant_question"],
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
        if "messages" not in st.session_state or st.session_state["messages"] == []:
            st.session_state["messages"] = [system_message]
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
            if "messages" in function_params:
                function_params["messages"] = st.session_state["messages"]
                logging.info(f"current_message_stream:{function_params['messages']}")
            st.write(function_name)
            st.write(function_params)
            if function_name == "stop_processing":
                final_question, st.session_state["messages"] = eval(
                    f"{function_name}(**{function_params})"
                )
                st.write(final_question)
                # Here, you would send `final_question` to your model
                break
            else:
                st.session_state["messages"] = eval(
                    f"{function_name}(**{function_params})"
                )
        # logging.info(f"current_message_stream:{st.write(st.session_state['messages'])}")
