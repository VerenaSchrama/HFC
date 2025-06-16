from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import os
from utils import reset_session, load_rag_chain, add_to_chat_history
from auth_service import AuthService
from profile_service import ProfileService
from config import (
    ERROR_MESSAGES, 
    SUCCESS_MESSAGES, 
    SESSION_TIMEOUT,
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPPORT_OPTIONS,
    DIETARY_OPTIONS,
    CYCLE_PHASES
)
import streamlit.components.v1 as components
import json
from fpdf import FPDF
import time
import logging

# More info & guidance page logic at the very top
if 'show_info_page' not in st.session_state:
    st.session_state['show_info_page'] = False

def show_info_page():
    st.title("More Info & Guidance")
    st.markdown("""
    ## Welcome to HerFoodCode!
    HerFoodCode is your scientific cycle nutrition assistant. This app helps you personalize your nutrition and wellness journey based on your menstrual cycle phase, support goals, and dietary preferences.

    ### How to Use the App
    1. **Personalize:** Fill in your current cycle phase, support goal, and dietary preferences.
    2. **Chat:** Ask questions or use the suggested questions to get science-backed nutrition advice tailored to your needs.
    3. **Download:** Download your recommendations as PDF or text.
    4. **Feedback:** Use the feedback box to share your thoughts or ask for more help.

    ### About HerFoodCode
    HerFoodCode is designed to empower you with knowledge and practical tips for every phase of your cycle. All recommendations are based on the latest nutritional science and are tailored to your unique needs.

    If you have any questions, feel free to use the chat or the feedback box!
    """)
    if st.button("Back to app", key="back_to_app_btn"):
        st.session_state['show_info_page'] = False
        st.rerun()

# Place the button at the very top of the sidebar, before any other sidebar code
top_sidebar_placeholder = st.sidebar.empty()
if top_sidebar_placeholder.button("More info & guidance", key="info_btn"):
    st.session_state['show_info_page'] = True
    st.rerun()

# Show info page if selected, otherwise show main app
if st.session_state.get('show_info_page', False):
    show_info_page()
    st.stop()

# Add a non-widget element at the very top to help prevent auto-scroll
st.markdown("\n")

# Always scroll to top on load/rerun (delayed for guaranteed fix)
st.markdown(
    """
    <script>
        setTimeout(function() {
            window.parent.document.documentElement.scrollTop = 0;
            window.parent.scrollTo(0, 0);
        }, 500);
    </script>
    """,
    unsafe_allow_html=True
)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

auth_service = AuthService()
profile_service = ProfileService()

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "guest_mode" not in st.session_state:
    st.session_state.guest_mode = False
if "personalization_completed" not in st.session_state:
    st.session_state.personalization_completed = False
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "last_activity" not in st.session_state:
    st.session_state.last_activity = datetime.now()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Suggested starter questions
suggested_questions = [
    "Give me a personal overview of foods for each of the 4 cycle phases.",
    "Review my previous meal choices and give me feedback.",
    "What foods are best for my current cycle phase?",
    "Give me a 3-day breakfast plan.",
    "Why is organic food important for my cycle?",
    "What nutritional seeds support my phase (seed syncing)?"
]

# Initialize session state variables for personalization and chat
if "phase" not in st.session_state:
    st.session_state.phase = None
if "support_goal" not in st.session_state:
    st.session_state.support_goal = ""
if "dietary_preferences" not in st.session_state:
    st.session_state.dietary_preferences = []

# Logo at the top
st.image("images/HerFoodCodeLOGO.png", width=120)

# Title
st.title("Your Scientific Cycle Nutrition Assistant")

query_params = st.query_params
logging.warning(f"RAW QUERY PARAMS: {query_params}")
if "token" in query_params and st.session_state.get("show_info_page", False) is False:
    token_param = query_params["token"]
    # Handle both list and string types
    if isinstance(token_param, list):
        token = token_param[0]
    else:
        token = token_param
    st.header("Email Verification")
    with st.spinner("Verifying your email..."):
        success, msg = auth_service.verify_email(token)
        if success:
            st.success("Verification successful! Welcome!")
            if st.button("Go to login page and get started"):
                st.experimental_set_query_params()
                st.rerun()
        else:
            st.error(f"Verification failed: {msg}")
    st.stop()
# --- END EMAIL VERIFICATION HANDLER ---

# Login/Register or Guest Access
if not st.session_state.logged_in and not st.session_state.guest_mode:
    st.write("Welcome! Choose how you'd like to proceed:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login or Register")
        auth_mode = st.radio("Select option", ["Login", "Register"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        # Show 'Forgot password?' button only in Login mode
        if auth_mode == "Login":
            if st.button("Forgot password?"):
                st.session_state.show_reset = True

        # Show password reset form
        if st.session_state.get("show_reset"):
            reset_email = st.text_input("Enter your email to reset password")
            if st.button("Send reset link"):
                success, msg = auth_service.send_password_reset(reset_email)
                if success:
                    st.success("Check your email for a reset link.")
                else:
                    st.error(msg)
            if st.button("Back to login/register"):
                st.session_state.show_reset = False
                st.rerun()
            st.stop()

        if auth_mode == "Register":
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Register"):
                if password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, msg = auth_service.register_user(email, password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
        else:
            if st.button("Login"):
                success, user_data, msg = auth_service.login_user(email, password)
                if success:
                    st.session_state.user_id = user_data["id"]
                    st.session_state.logged_in = True
                    st.session_state.login_attempts = 0
                    st.rerun()
                else:
                    st.session_state.login_attempts += 1
                    st.error(msg)
    
    with col2:
        st.subheader("Try as Guest")
        st.write("Experience the chatbot without creating an account")
        if st.button("Continue as Guest"):
            st.session_state.guest_mode = True
            st.rerun()
    
    st.stop()

# Handle password reset via token in URL
query_params = st.query_params
if "token" in query_params:
    token = query_params["token"][0]
    st.header("Reset Your Password")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm New Password", type="password")
    if st.button("Reset Password"):
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, msg = auth_service.reset_password(token, new_password)
            if success:
                st.success("Password reset successful! You can now log in.")
            else:
                st.error(msg)
    st.stop()

# Personalization
st.header("Personalization")

# Support goal and dietary preferences FIRST
st.session_state.support_goal = st.selectbox("Support goal", ["Select..."] + SUPPORT_OPTIONS)
st.session_state.dietary_preferences = st.multiselect("Dietary preferences", DIETARY_OPTIONS)

# Add divider before Option 1
st.markdown("---")

# Manual override first
st.markdown("#### Option 1: Choose your current cycle phase manually ⬇️")
phase_override = st.selectbox(
    "Select your current cycle phase (optional)",
    ["Select..."] + ["Menstrual", "Follicular", "Ovulatory", "Luteal"],
    index=0,
    label_visibility="collapsed"
)

# Add intro for auto detection (no divider here)
st.markdown("#### Option 2: Or let me help you to find your current phase:")

has_cycle = st.radio("Do you have a (regular) menstrual cycle?", ("Yes", "No"))

if has_cycle == "Yes":
    today = datetime.now().date()
    st.session_state.second_last_period = st.date_input("Second most recent period start date", value=today)
    st.session_state.last_period = st.date_input("Most recent period start date", value=today)

    if st.session_state.last_period and st.session_state.second_last_period:
        if st.session_state.second_last_period > st.session_state.last_period:
            st.session_state.second_last_period, st.session_state.last_period = st.session_state.last_period, st.session_state.second_last_period

        if st.session_state.last_period != today and st.session_state.second_last_period != today:
            cycle_length = (st.session_state.last_period - st.session_state.second_last_period).days
            if cycle_length <= 10:
                st.error("Your periods seem too close together. Please check the entered dates.")
            else:
                st.session_state.cycle_length = cycle_length
                days_since_last = (today - st.session_state.last_period).days

                if days_since_last <= 5:
                    detected_phase = "Menstrual"
                elif days_since_last <= 14:
                    detected_phase = "Follicular"
                elif days_since_last <= 21:
                    detected_phase = "Ovulatory"
                else:
                    detected_phase = "Luteal"

                st.session_state.phase = phase_override if phase_override else detected_phase
                if not phase_override:
                    st.success(f"Based on your data, you are likely in the **{st.session_state.phase}** phase.")
                st.session_state.personalization_completed = True
else:
    st.subheader("No active menstrual cycle detected.")
    pseudo_choice = st.radio("Would you like:", ("Get general energetic advice", "Start with a pseudo-cycle based on a 28-day rhythm"))

    if pseudo_choice:
        if pseudo_choice == "🌿 Get general energetic advice":
            st.session_state.phase = "General"
        else:
            st.session_state.phase = "Menstrual"
            st.session_state.cycle_length = 28
        st.success(f"Selected: {pseudo_choice}")
        st.session_state.personalization_completed = True

# Manual override always takes precedence if selected
if phase_override and phase_override in ["Menstrual", "Follicular", "Ovulatory", "Luteal"]:
    st.session_state.phase = phase_override
    st.success(f"You selected: **{phase_override}** phase manually.")
    st.session_state.personalization_completed = True

if st.session_state.phase and st.session_state.support_goal and st.session_state.dietary_preferences:
    st.session_state.personalization_completed = True

# --- Chat area: chat bubbles with speaker labels ---
st.markdown('''
<style>
.chat-bubble-user {
    background: #2d2d2d;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    margin-left: 20%;
    margin-right: 0;
    text-align: right;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.chat-bubble-assistant {
    background: #232323;
    color: #fff;
    border-radius: 16px 16px 16px 4px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    margin-right: 20%;
    margin-left: 0;
    text-align: left;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.speaker-label {
    font-weight: bold;
    font-size: 0.95em;
    margin-bottom: 0.2em;
    color: #e07a5f;
}
</style>
''', unsafe_allow_html=True)

if st.session_state.get("personalization_completed"):
    st.header("Chat History")
    if st.session_state.chat_history:
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'''<div class="chat-bubble-user"><div class="speaker-label">You</div>{msg}</div>''', unsafe_allow_html=True)
            else:
                st.markdown(f'''<div class="chat-bubble-assistant"><div class="speaker-label">Assistant</div>{msg}</div>''', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#888; margin:2em 0; text-align:center;">Start the conversation by asking your first question below!</div>', unsafe_allow_html=True)

# Input at the bottom, always visible after latest message
if st.session_state.get("clear_chat_input"):
    st.session_state["chat_input"] = ""
    st.session_state["clear_chat_input"] = False

# --- Use Streamlit's st.chat_input for always-visible chat input ---
user_question = st.chat_input("Type your question...")
if user_question:
    try:
        qa_chain = load_rag_chain()

        # 🔁 Voeg gepersonaliseerde context toe
        enriched_question = f"""
        {user_question}

        My current cycle phase is: {st.session_state.get("phase", "not provided")}.
        My current goal is: {st.session_state.get("support_goal", "not provided")}.
        My dietary preferences are: {', '.join(st.session_state.get("dietary_preferences", [])) or "not provided"}.
        """

        response  = qa_chain({
            "question": enriched_question
        })["answer"]

        add_to_chat_history("user", user_question)
        add_to_chat_history("assistant", response)
        st.rerun()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Logout button for logged-in users
if st.session_state.logged_in:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.personalization_completed = False
        st.session_state.chat_history = []
        st.rerun()

# --- Always-Visible Personalization Summary in Sidebar ---
st.sidebar.markdown("## Your Personalization Summary")
if st.session_state.get("phase"):
    st.sidebar.markdown(f"**Cycle phase:** {st.session_state.phase}")
else:
    st.sidebar.markdown("**Cycle phase:** _Not set_")
if st.session_state.get("support_goal"):
    st.sidebar.markdown(f"**Support goal:** {st.session_state.support_goal}")
else:
    st.sidebar.markdown("**Support goal:** _Not set_")
if st.session_state.get("dietary_preferences"):
    st.sidebar.markdown(f"**Dietary preferences:** {', '.join(st.session_state.dietary_preferences)}")
else:
    st.sidebar.markdown("**Dietary preferences:** _None_")

# Divider between summary and suggested questions
st.sidebar.markdown("---")

# --- Suggested Questions Panel in Sidebar ---
suggested_questions = [
    "Give me a personal overview of foods for each of the 4 cycle phases to start experimenting with.",
    "Review my previous meal choices and give me feedback.",
    "What foods are best for my current cycle phase?",
    "Give me a 3-day breakfast plan.",
    "Why is organic food important for my cycle?",
    "What nutritional seeds support my phase (seed syncing)?"
]
st.sidebar.markdown("## 💡 Suggested Questions")
for i, question in enumerate(suggested_questions):
    if st.sidebar.button(question, key=f"sidebar_suggested_q_{i}"):
        add_to_chat_history("user", question)

        if question == "Review my previous meal choices and give me feedback.":
            response = "Please log your meals in the following format: Day + Meal ingredients."
            add_to_chat_history("assistant", response)
            st.rerun()
        else:
            try:
                qa_chain = load_rag_chain()

                enriched_question = f"""
                {question}

                My current cycle phase is: {st.session_state.get("phase", "not provided")}.
                My current goal is: {st.session_state.get("support_goal", "not provided")}.
                My dietary preferences are: {', '.join(st.session_state.get("dietary_preferences", [])) or "not provided"}.
                """

                response = qa_chain({
                    "question": enriched_question
                })["answer"]

                add_to_chat_history("assistant", response)

                if i == 0:
                    st.session_state["recommendations_response"] = response

                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Divider between suggested questions and feedback
st.sidebar.markdown("---")

# --- Feedback Box at the bottom of the sidebar ---
if st.session_state.get("clear_feedback_text"):
    st.session_state["feedback_text"] = ""
    st.session_state["clear_feedback_text"] = False
st.sidebar.markdown("## Feedback")
feedback_text = st.sidebar.text_area("Have feedback or a question I didn't answer?", key="feedback_text")
if st.sidebar.button("Submit Feedback", key="submit_feedback"):
    if feedback_text.strip():
        feedback_data = {
            "user_id": st.session_state.get("user_id", "guest"),
            "timestamp": datetime.utcnow().isoformat(),
            "feedback": feedback_text.strip()
        }
        try:
            supabase.table("feedback").insert(feedback_data).execute()
            st.sidebar.success("Thank you for your feedback!")
            st.session_state["clear_feedback_text"] = True
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error submitting feedback: {str(e)}")
    else:
        st.sidebar.warning("Please enter your feedback before submitting.")

# Divider and Exit Guest Mode button at the bottom of the sidebar
if st.session_state.guest_mode:
    st.sidebar.markdown("---")
    if st.sidebar.button("Exit Guest Mode", key="sidebar_exit_guest"):
        st.session_state.guest_mode = False
        st.session_state.personalization_completed = False
        st.session_state.chat_history = []
        st.rerun()

# After rendering chat bubbles, show download if available
def recommendations_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    # Add logo (centered)
    logo_path = "images/HerFoodCodeLOGO.png"
    pdf.image(logo_path, x=pdf.w/2-15, y=10, w=30)
    pdf.ln(25)
    # Title with color #442369
    pdf.set_text_color(68, 35, 105)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Your Nutritional overview per cycle phase", ln=True, align='C')
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        # Make section headers (lines starting with a number and dot) colored
        if line.strip().startswith(tuple(str(i)+'.' for i in range(1,10))):
            pdf.set_text_color(68, 35, 105)
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 10, line)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=12)
        else:
            pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode('latin-1')

if st.session_state.get("recommendations_response"):
    st.markdown("### Download your recommendations")
    st.download_button(
        label="Download as PDF",
        data=recommendations_to_pdf(st.session_state["recommendations_response"]),
        file_name="cycle_phase_recommendations.pdf",
        mime="application/pdf"
    )
    st.download_button(
        label="Download as Text",
        data=st.session_state["recommendations_response"],
        file_name="cycle_phase_recommendations.txt",
        mime="text/plain"
    )

if not st.session_state.get("personalization_completed"):
    st.info("Please complete personalization above.")
    st.stop()