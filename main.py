import streamlit as st
import openai
import json
import numpy as np
import logging
from streamlit.runtime.scriptrunner import StopException, RerunException

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


if __name__ == "__main__":
    if st.session_state.get('rerun', False):
        st.session_state['rerun'] = False
        st.rerun()

if "knowledge_graph" not in st.session_state:
    st.session_state["knowledge_graph"] = {}

st.title("Finance Domain Chat Assistant")

def reset_knowledge_graph():
    st.session_state["knowledge_graph"] = {}
    st.session_state['rerun'] = True
    st.stop()

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
The table schemas below are provided in the format - table_name ( column_name_1 data_type_1, column_name_2 data_type_2, ... ).
    - journal ( posting_date str, fiscal_year str, fiscal_quarter str, fiscal_month str, fiscal_period str, fiscal_year_quarter str, fiscal_year_month str, fiscal_year_period str, account_number str, account_name str, account_type str, account_category str, income_statement_group str, amount decimal(18,2), department_number, department_name, cost_center_number str, cost_center_name str, profit_center_number str, profit_center_name str, purchase_order_number str, supplier_number str, supplier_name str, material_number str, material_name str, material_group_number str, material_group_name str, sales_order_number str, customer_number str, customer_name str, product_number str, product_name str, product_group_number str, product_group_name str, transaction_id str, transaction_type str, transaction_document_number str, transaction_document_item str, region).
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
    region is a specific location that we might be interested in. 

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
                            "description": 'These will only be Specifc jargon words that we have disambiguated with user inputs. Only include terms that are uncommon. Should be case insensitive while adding a knowledge pieces. Do not add repeat jargon words in the knowledge graph. Return {} when there is nothing new to add.',
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
You are a helpful AI assistant specialized in query refinement and summarization. Your task is to analyze the given context and generate a single, well-defined question that perfectly encapsulates the essence of the context. This question should be relevant to the predetermined scope.
Instructions:

Carefully review the provided context.
Determine if the context is relevant to the predefined scope.
If relevant, summarize the context into a single, comprehensive question.
Utilize your knowledge base to substitute terms with their most relevant and precise meanings.
Ensure the question specifies the type of entity along with its name, when applicable.
Format your response as "Question: [Your refined question]"
Only provide the final refined question. Do not include any answers or explanations.

If the context is not relevant to the predefined scope, simply respond with "The provided context is not relevant to the specified scope."
Remember, your goal is to create a clear, concise, and well-formed query that captures the essence of the given context while adhering to the specified guidelines.
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
            st.session_state['rerun'] = True
            st.stop()

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
            st.session_state["follow_up_options"] = None
            st.session_state['rerun'] = True  # Reset options after use
            st.stop()
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
                st.session_state['rerun'] = True
                st.stop()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            st.error("An error occurred. Please try again.")


