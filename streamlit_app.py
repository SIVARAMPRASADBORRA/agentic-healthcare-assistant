"""Agentic Healthcare Assistant.

Run with:  streamlit run streamlit_app.py
Requires:  a .env file with OPENAI_API_KEY (same as the notebook), and
           records.xlsx / vector_store / eval_log.csv produced by the notebook
           to already exist in this folder (run the notebook once first).
"""
import os
import re
import json
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared secrets live in a single .env under the "Agentic AI" workspace root
# (Sivaram folder/.env) so every project draws from one place instead of each
# needing its own copy. A project-local .env, if present, is loaded first and
# takes precedence for any key it defines.
GLOBAL_ENV_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "Sivaram folder", ".env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(GLOBAL_ENV_FILE)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.prompts import PromptTemplate

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
RECORDS_FILE = os.path.join(BASE_DIR, "records.xlsx")
APPOINTMENTS_FILE = os.path.join(BASE_DIR, "appointments.xlsx")
EVAL_LOG_FILE = os.path.join(BASE_DIR, "eval_log.csv")
PATIENT_COLUMNS = ["Phone_number", "Email", "Name", "Age", "Gender", "Address", "Summary"]
APPOINTMENT_COLUMNS = [
    "Phone_number",
    "Patient_Name",
    "Specialty_Or_Test",
    "Doctor",
    "Appointment_Date",
    "Appointment_Time",
    "Token",
    "Booked_At",
]
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🏥", layout="wide")


@st.cache_resource
def get_llm():
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.1, max_tokens=2000)


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def normalize_phone(number) -> str:
    digits = re.sub(r"\D", "", str(number))
    return digits[-10:] if len(digits) >= 10 else digits


def read_patients() -> pd.DataFrame:
    if not os.path.exists(RECORDS_FILE):
        return pd.DataFrame(columns=PATIENT_COLUMNS)
    df = pd.read_excel(RECORDS_FILE)
    for col in PATIENT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def get_patient_by_phone(phone: str) -> Optional[Dict]:
    df = read_patients()
    if df.empty:
        return None
    df["_norm"] = df["Phone_number"].apply(normalize_phone)
    match = df[df["_norm"] == normalize_phone(phone)]
    if match.empty:
        return None
    return match.iloc[-1].drop(labels=["_norm"]).to_dict()


def save_patient(details: Dict) -> None:
    """Append a newly registered patient to the records store."""
    df = read_patients()
    new_row = {col: details.get(col) for col in PATIENT_COLUMNS}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(RECORDS_FILE, index=False)


def read_appointments() -> pd.DataFrame:
    if not os.path.exists(APPOINTMENTS_FILE):
        return pd.DataFrame(columns=APPOINTMENT_COLUMNS)
    df = pd.read_excel(APPOINTMENTS_FILE)
    for col in APPOINTMENT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def save_appointment(row: Dict) -> None:
    """Append a booked appointment so repeat visits each create a new record."""
    df = read_appointments()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(APPOINTMENTS_FILE, index=False)


DOCTOR_DIRECTORY = {
    "nephrologist": ["Dr. Priya Sharma (Nephrology)", "Dr. Rajesh Kumar (Nephrology)"],
    "cardiologist": ["Dr. Amit Patel (Cardiology)", "Dr. Sneha Reddy (Cardiology)"],
    "dermatologist": ["Dr. Michael Chen (Dermatology)", "Dr. Lisa Park (Dermatology)"],
    "endocrinologist": ["Dr. Kavita Rao (Endocrinology)", "Dr. Steven Lee (Endocrinology)"],
    "general": ["Dr. John Smith (General Medicine)", "Dr. Sarah Johnson (General Medicine)"],
}


def book_appointment_or_test(
    test_name: str, phone_number: Optional[str] = None, patient_name: Optional[str] = None
) -> str:
    today = datetime.now()
    appointment_date = (today + timedelta(days=random.randint(0, 6))).replace(
        hour=random.randint(9, 17), minute=0, second=0, microsecond=0
    )
    token = random.randint(100000, 999999)
    doctor_list = DOCTOR_DIRECTORY["general"]
    for key, doctors in DOCTOR_DIRECTORY.items():
        if key in test_name.lower():
            doctor_list = doctors
            break
    doctor = random.choice(doctor_list)

    save_appointment(
        {
            "Phone_number": phone_number or "unknown",
            "Patient_Name": patient_name or "Unknown",
            "Specialty_Or_Test": test_name,
            "Doctor": doctor,
            "Appointment_Date": appointment_date.strftime("%Y-%m-%d"),
            "Appointment_Time": appointment_date.strftime("%H:%M"),
            "Token": token,
            "Booked_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    return (
        f"Appointment for '{test_name}' booked successfully.\n"
        f"Doctor: {doctor}\n"
        f"Date: {appointment_date.strftime('%A, %B %d, %Y')}\n"
        f"Time: {appointment_date.strftime('%I:%M %p')}\n"
        f"Token Number: {token}"
    )


def get_patient_context(question: str, patient_key: Optional[str]) -> str:
    if not patient_key:
        return "No patient identified."
    index_path = os.path.join(VECTOR_DIR, patient_key)
    if not os.path.exists(index_path):
        return "No indexed report for this patient."
    store = FAISS.load_local(index_path, get_embeddings(), allow_dangerous_deserialization=True)
    docs = store.as_retriever(search_kwargs={"k": 5}).invoke(question)
    return "\n\n".join(d.page_content for d in docs) if docs else "No relevant info found."


def search_medical_info(topic: str) -> str:
    try:
        from langchain_community.tools.pubmed.tool import PubmedQueryRun
        result = PubmedQueryRun().run(topic)
        return (result or "No results.")[:1500]
    except Exception as e:
        return f"Search failed: {e}"


PLANNER_PROMPT = PromptTemplate(
    template="""Break the query into subtasks mapped to tools:
get_patient_context, book_appointment_or_test, search_medical_info.
Patient known: {patient_known}
Query: "{query}"
Return ONLY JSON: {{"intent": "...", "steps": [{{"tool": "...", "tool_input": "..."}}]}}
""",
    input_variables=["patient_known", "query"],
)

SYNTHESIS_PROMPT = PromptTemplate(
    template="""You are an AI Healthcare Assistant. Be empathetic and concise.
Query: {query}
Patient: {patient_info}
Tool results: {tool_results}
Write the final response. Confirm any bookings clearly.
""",
    input_variables=["query", "patient_info", "tool_results"],
)


class AgentState(TypedDict):
    query: str
    phone_number: Optional[str]
    patient_info: Optional[Dict]
    plan: List[Dict]
    tool_results: List[Dict]
    response: str
    history: List[Dict]


@st.cache_resource
def build_graph():
    llm = get_llm()

    def identify_patient_node(state):
        phone = state.get("phone_number")
        state["patient_info"] = get_patient_by_phone(phone) if phone else None
        return state

    def planner_node(state):
        prompt = PLANNER_PROMPT.format(patient_known=bool(state.get("patient_info")), query=state["query"])
        resp = llm.invoke(prompt)
        cleaned = re.sub(r"```(?:json)?", "", resp.content).strip()
        try:
            state["plan"] = json.loads(cleaned).get("steps", [])
        except Exception:
            state["plan"] = []
        return state

    def executor_node(state):
        phone_key = normalize_phone(state["phone_number"]) if state.get("phone_number") else None
        patient_name = (state.get("patient_info") or {}).get("Name")
        results = []
        for step in state.get("plan", []):
            name, tool_input = step.get("tool"), step.get("tool_input", "")
            if name == "get_patient_context":
                out = get_patient_context(tool_input, phone_key)
            elif name == "book_appointment_or_test":
                out = book_appointment_or_test(tool_input, phone_number=phone_key, patient_name=patient_name)
            elif name == "search_medical_info":
                out = search_medical_info(tool_input)
            else:
                out = f"Unknown tool: {name}"
            results.append({"tool": name, "input": tool_input, "output": out})
        state["tool_results"] = results
        return state

    def synthesizer_node(state):
        tool_text = "\n\n".join(f"[{r['tool']}] {r['output']}" for r in state.get("tool_results", [])) or "(none)"
        prompt = SYNTHESIS_PROMPT.format(
            query=state["query"],
            patient_info=json.dumps(state.get("patient_info"), default=str),
            tool_results=tool_text,
        )
        resp = llm.invoke(prompt)
        state["response"] = resp.content
        hist = state.get("history", [])
        hist.append({"role": "user", "content": state["query"]})
        hist.append({"role": "assistant", "content": resp.content})
        state["history"] = hist
        return state

    gb = StateGraph(AgentState)
    gb.add_node("identify_patient", identify_patient_node)
    gb.add_node("plan", planner_node)
    gb.add_node("execute", executor_node)
    gb.add_node("synthesize", synthesizer_node)
    gb.set_entry_point("identify_patient")
    gb.add_edge("identify_patient", "plan")
    gb.add_edge("plan", "execute")
    gb.add_edge("execute", "synthesize")
    gb.add_edge("synthesize", END)
    return gb.compile(checkpointer=MemorySaver())


agent = build_graph()

# ---------------------------------------------------------------------- UI --
st.title("🏥 Agentic Healthcare Assistant")

tab_chat, tab_patient, tab_eval, tab_memory = st.tabs(
    ["💬 Chat & Booking", "📋 Patient / Doctor View", "📊 Evaluation Metrics", "🧠 Memory & Plan Trace"]
)

with st.sidebar:
    st.header("Patient")
    phone_input = st.text_input("Phone number", value=st.session_state.get("phone", ""))

    if st.button("Load patient"):
        st.session_state["phone"] = phone_input
        found = get_patient_by_phone(phone_input) if phone_input else None
        st.session_state["patient"] = found
        st.session_state["show_new_patient_form"] = bool(phone_input) and not found
        # Switching patients (including a brand-new one) starts a clean chat.
        st.session_state["messages"] = []
        st.session_state["last_result"] = None
        st.session_state["chat_session_id"] = st.session_state.get("chat_session_id", 0) + 1

    patient = st.session_state.get("patient")
    if patient:
        st.success(f"Loaded: {patient.get('Name')}")
    elif st.session_state.get("show_new_patient_form"):
        st.warning("No patient found with that number. Register below.")

    if st.session_state.get("show_new_patient_form"):
        with st.form("new_patient_form"):
            st.subheader("Register new patient")
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            email = st.text_input("Email")
            address = st.text_input("Address")
            submitted = st.form_submit_button("Save new patient")
            if submitted:
                if not name:
                    st.error("Name is required.")
                else:
                    new_patient = {
                        "Phone_number": st.session_state["phone"],
                        "Email": email,
                        "Name": name,
                        "Age": age,
                        "Gender": gender,
                        "Address": address,
                        "Summary": "",
                    }
                    save_patient(new_patient)
                    st.session_state["patient"] = new_patient
                    st.session_state["show_new_patient_form"] = False
                    st.session_state["messages"] = []
                    st.session_state["last_result"] = None
                    st.session_state["chat_session_id"] = st.session_state.get("chat_session_id", 0) + 1
                    st.success(f"Patient {name} registered.")
                    st.rerun()

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = 0

    header_col, clear_col = st.columns([5, 1])
    with clear_col:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_result"] = None
            # Bump the thread id so the agent's checkpointed memory starts fresh too.
            st.session_state["chat_session_id"] += 1
            st.rerun()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about a diagnosis, book an appointment, or request medical info..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                thread_key = normalize_phone(st.session_state.get("phone", "")) or "anonymous"
                config = {
                    "configurable": {
                        "thread_id": f"{thread_key}_{st.session_state['chat_session_id']}"
                    }
                }
                result = agent.invoke({"query": prompt, "phone_number": st.session_state.get("phone")}, config=config)
                st.markdown(result["response"])
                st.session_state["last_result"] = result
        st.session_state["messages"].append({"role": "assistant", "content": result["response"]})

with tab_patient:
    st.subheader("All patients")
    st.dataframe(read_patients(), use_container_width=True)

    st.subheader("All appointments")
    st.dataframe(read_appointments(), use_container_width=True)

    current_patient = st.session_state.get("patient")
    current_phone = st.session_state.get("phone")
    if current_patient and current_phone:
        st.subheader(f"Appointment history — {current_patient.get('Name')}")
        appts = read_appointments()
        if not appts.empty:
            appts["_norm"] = appts["Phone_number"].apply(normalize_phone)
            history = appts[appts["_norm"] == normalize_phone(current_phone)].drop(columns=["_norm"])
        else:
            history = appts
        if history.empty:
            st.info("No appointments booked yet for this patient.")
        else:
            st.dataframe(history, use_container_width=True)

with tab_eval:
    st.subheader("Model evaluation log")
    if os.path.exists(EVAL_LOG_FILE):
        eval_df = pd.read_csv(EVAL_LOG_FILE)
        st.dataframe(eval_df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg relevance", round(eval_df["relevance"].mean(), 2))
        col2.metric("Avg accuracy", round(eval_df["accuracy"].mean(), 2))
        col3.metric("Booking accuracy", f"{(eval_df['booking_correct'].mean() * 100):.0f}%")
        st.bar_chart(eval_df[["relevance", "accuracy"]])
    else:
        st.info("Run the evaluation cell in the notebook to generate eval_log.csv.")

with tab_memory:
    st.subheader("Latest plan + tool trace")
    last = st.session_state.get("last_result")
    if last:
        st.json(last.get("plan", []))
        st.write("Tool results:")
        st.json(last.get("tool_results", []))
        st.write("Conversation history for this patient thread:")
        st.json(last.get("history", []))
    else:
        st.info("Ask something in the Chat tab to see the plan/memory trace here.")
