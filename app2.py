import streamlit as st
import time
import random
import requests

GEMINI_API_KEY = "AIzaSyBh2B-y_1cyDXPA3xc7zPoBuf1yaRXs28k"
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

mental_health_keywords = [
    # الأساسيات
    "depression", "anxiety", "stress", "mental health", "therapy",
    "sad", "worried", "panic", "emotion", "feeling", "psychologist",
    "mental illness", "mental disorder", "mind", "brain", "ptsd", "trauma",
    "suicide", "psychotherapy", "counseling", "mental wellbeing",
    "self-esteem", "mood", "bipolar", "schizophrenia", "addiction",
    "insomnia", "fatigue", "anger", "grief", "lonely", "fear", "help",
    "nervous", "panic attack", "overwhelmed", "therapy session", "coping",
    "emotional pain", "breakdown", "support", "recovery", "healing", "sleep", "awake"

    # Anxiety
    "anxious", "nervousness", "worry", "phobia", "social anxiety",
    "panic attack", "heart racing", "overthinking", "restless",
    "uneasy", "dread", "apprehensive", "can't relax", "tight chest",
    "breathless", "sweating", "shaking", "hyperventilation",
    "anticipatory anxiety", "fear of failure", "fear of judgment",
    "fear of people", "fear of speaking", "agoraphobia",

    # Depression
    "hopeless", "helpless", "worthless", "empty", "crying", "tearful",
    "down", "low mood", "miserable", "loss of interest", "no energy",
    "can't focus", "guilt", "shame", "isolation", "withdrawal",
    "self-harm", "suicidal", "dark thoughts", "mental pain", "no motivation",
    "can't get out of bed", "cry for help", "lost interest", "numb",
    "feeling like a burden", "negative thoughts",

    # Bipolar Disorder
    "mania", "manic", "hypomania", "euphoria", "high energy",
    "racing thoughts", "impulsive", "grandiosity", "reckless",
    "talkative", "insomnia", "hyperactive", "irritable", "agitation",
    "mood swings", "emotional highs", "emotional lows", "depressive",
    "manic episode", "mixed episode", "rapid cycling",

    # PTSD
    "post-traumatic stress", "flashbacks", "nightmares", "traumatic event",
    "triggers", "startle response", "hypervigilance", "avoidance",
    "emotional detachment", "feeling unsafe", "memories", "abuse survivor",
    "war trauma", "accident survivor", "violence", "survivor guilt",

    # OCD
    "OCD", "obsessive", "compulsive", "repetitive behavior", "checking",
    "cleaning", "handwashing", "counting", "rituals", "germs",
    "contamination fear", "intrusive thoughts", "mental compulsions",
    "order", "perfectionism", "symmetry", "can't stop thinking", "obsession",

    # Eating Disorders
    "eating disorder", "anorexia", "bulimia", "binge eating", "purging",
    "vomiting", "calorie counting", "body image", "thinspiration",
    "weight loss", "not eating", "restricting", "fat", "feel fat",
    "mirror", "scale", "dieting", "starving", "overeating",
    "guilt after eating", "control",

    # Schizophrenia
    "psychosis", "hallucination", "delusion", "paranoia", "voices",
    "seeing things", "hearing voices", "disorganized thoughts", "catatonia",
    "mental break", "thought disorder", "delusional", "psychotic episode",

    # Burnout
    "burnout", "mental exhaustion", "emotional exhaustion", "drained",
    "demotivated", "can’t do this anymore", "breakdown", "overworked",
    "too much stress", "burned out", "need a break", "pressure", "mental strain",

    # إضافات كلمات منفردة
    "overthinking", "stressed", "tired", "exhausted", "fragile", "numbness",
    "detached", "confused", "lost", "headache", "stomachache", "chestpain",
    "dizziness", "trembling", "muscletension", "loneliness", "rejection",
    "abandonment", "betrayal", "disappointment", "hopelessness", "panic",
    "venting", "medication", "antidepressants", "anxiolytics", "mindfulness",
    "meditation", "journaling", "selfcare", "relaxation", "breathing",
    "judgment", "stigma", "embarrassment", "misunderstood", "taboo",
    "crisis", "emergency", "nightsweats", "nightterrors", "waking",
    "sideeffects", "drowsiness", "weightgain", "drymouth", "blurredvision",
    "hope", "progress", "stronger", "growth", "resilience", "compassion", "feel"
]


def is_mental_health_related(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in mental_health_keywords)

import random

def generate_gemini_response(prompt: str) -> str:
    prompt_lower = prompt.lower()

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    thanks = ["thank you", "thanks", "thx", "thank you very much", "thanks a lot"]
    goodbyes = ["bye", "goodbye", "see you", "see ya", "take care", "farewell", "later", "catch you later", "talk to you later", "have a nice day", "have a good day", "see you soon"]

    greeting_responses = [
        "Hello! I'm MentalEase, your mental health assistant. How can I support you today?",
        "Hi there! MentalEase at your service. How are you feeling?",
        "Hey! I'm here to listen and help you with anything related to mental health.",
        "Good to see you! How can I assist with your mental wellbeing today?"
    ]

    thanks_responses = [
        "You're very welcome! I'm here whenever you need support.",
        "No problem! Glad I could help.",
        "Anytime! Feel free to reach out whenever you want.",
        "Happy to assist! Take care of yourself."
    ]

    goodbye_responses = [
        "Take care! Remember, I'm here whenever you need to talk.",
        "Goodbye! Wishing you the best for your mental health.",
        "See you later! Stay strong and take care.",
        "Farewell! Don't hesitate to come back if you need support."
    ]

    # الردود الترحيبية
    if any(greet in prompt_lower for greet in greetings):
        return random.choice(greeting_responses)

    # الردود على الشكر
    if any(thank in prompt_lower for thank in thanks):
        return random.choice(thanks_responses)

    # الردود على الوداع
    if any(bye in prompt_lower for bye in goodbyes):
        return random.choice(goodbye_responses)

 
    # لو الرسالة مش متعلقة بالصحة النفسية
    if not is_mental_health_related(prompt):
        return (
            "I'm here to support you with mental health topics only. "
            "If you want to talk about feelings, stress, or anything related to mental wellbeing, please share. "
            "Otherwise, I'm sorry, I can't help with that."
        )
    
    # لو الموضوع متعلق بالصحة النفسية، نرسل طلب لـ Gemini API مع تعليمات دعم نفسي
    headers = {"Content-Type": "application/json"}
    modified_prompt = (
        f"You are a compassionate and supportive mental health assistant. "
        f"Respond empathetically and encouragingly, using 2 to 50 words:\n\n{prompt}"
    )
    payload = {
        "contents": [
            {
                "parts": [{"text": modified_prompt}]
            }
        ]
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()
        reply = content["candidates"][0]["content"]["parts"][0]["text"]
        words = reply.split()
        if len(words) > 50:
            reply = " ".join(words[:50]) + "..."
        return reply.strip()
    except Exception:
        return (
            "I'm sorry, I'm having trouble responding right now. "
            "Please try again later or consider talking to a mental health professional."
        )


st.set_page_config(page_title="MentalEase", page_icon="🤖")

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "waiting_for_bot" not in st.session_state:
    st.session_state.waiting_for_bot = False
if "streaming_user" not in st.session_state:
    st.session_state.streaming_user = False
if "streaming_bot" not in st.session_state:
    st.session_state.streaming_bot = False
if "user_input_temp" not in st.session_state:
    st.session_state.user_input_temp = ""
if "bot_response_stream" not in st.session_state:
    st.session_state.bot_response_stream = ""
if "user_streamed_text" not in st.session_state:
    st.session_state.user_streamed_text = ""

def start_new_chat():
    new_chat = {"title": f"Chat {len(st.session_state.chat_sessions) + 1}", "history": []}
    st.session_state.chat_sessions.append(new_chat)
    st.session_state.current_chat = len(st.session_state.chat_sessions) - 1
    st.session_state.waiting_for_bot = False
    st.session_state.user_input_temp = ""
    st.session_state.bot_response_stream = ""
    st.session_state.user_streamed_text = ""

# Sidebar for chats and new chat button

st.sidebar.title("Chats")
if st.sidebar.button("➕ New Chat"):
    start_new_chat()

if st.session_state.chat_sessions:
    for i, chat in enumerate(st.session_state.chat_sessions):
        if st.sidebar.button(chat["title"], key=f"chat_{i}"):
            st.session_state.current_chat = i
            st.session_state.waiting_for_bot = False
            st.session_state.user_input_temp = ""
            st.session_state.bot_response_stream = ""
            st.session_state.user_streamed_text = ""
else:
    st.sidebar.write("No chats yet. Click **New Chat** to start.")

if st.session_state.current_chat is None:
    start_new_chat()


st.markdown(
    """
    <h1 style="font-weight: bold; font-size: 2rem;">
        <span style="animation: flash 2.0s infinite;">💬</span>
        <span style="color: gray;">MentalEase</span>: 
        <span style="color: beige;">Mental Health Detection Chatbot</span>
    </h1>

    <style>
    @keyframes flash {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

chat_history = st.session_state.chat_sessions[st.session_state.current_chat]["history"]

# عرض المحادثة كاملة
for msg in chat_history:
    role = msg["role"]
    content = msg["content"]
    color = "orange" if role == "You" else "gray"
    font = "Courier New" if role == "You" else "Georgia"
    label = "You" if role == "You" else "MentalEase"
    st.markdown(
        f"<b style='color: {color}; font-family: Arial, sans-serif;'>{label}:</b> "
        f"<span style='font-family: {font}, serif; font-size: 16px;'>{content}</span>",
        unsafe_allow_html=True
    )

user_display = st.empty()
bot_display = st.empty()
typing_placeholder = st.empty()

# نبدأ بانتظار إدخال المستخدم فقط لو مش شغال streaming
if not st.session_state.streaming_user and not st.session_state.waiting_for_bot:
    user_input = st.chat_input("I'm here to support you, how can I help you today?", key="user_input")
    if user_input:
        # شغلنا streaming للسؤال
        st.session_state.streaming_user = True
        st.session_state.user_input_temp = user_input
        st.session_state.user_streamed_text = ""
        st.session_state.waiting_for_bot = False

# Streaming السؤال (الرسالة من المستخدم) كلمة كلمة
if st.session_state.streaming_user:
    words = st.session_state.user_input_temp.split()
    for i in range(len(words)):
        st.session_state.user_streamed_text = " ".join(words[:i+1])
        user_display.markdown(
            f"<b style='color: orange; font-family: Arial, sans-serif;'>You:</b> "
            f"<span style='font-family: Courier New, monospace; font-size: 16px;'>{st.session_state.user_streamed_text}</span>",
            unsafe_allow_html=True
        )
        time.sleep(0.04)
    # بعد ما خلصنا نوقف streaming المستخدم ونبدأ انتظار الرد من البوت
    st.session_state.streaming_user = False
    st.session_state.waiting_for_bot = True
    st.session_state.bot_response_stream = ""
    chat_history.append({"role": "You", "content": st.session_state.user_input_temp})

# لو بننتظر البوت نعرض تأثير typing مع نقاط متحركة
if st.session_state.waiting_for_bot:
    dots_count = 0
    max_dots = 4
    full_reply = generate_gemini_response(st.session_state.user_input_temp)
    reply_words = full_reply.split()
    reply_so_far = ""

    for i in range(len(reply_words)):
        # عرض typing مع النقاط
        for dot_i in range(10):  # مدة مؤقت النقاط المتحركة قبل ظهور كل كلمة
            dots_count = (dots_count % max_dots) + 1
            dots = "." * dots_count
            typing_placeholder.markdown(
                f"<span style='font-family: Georgia, serif; font-size: 16px; color: gray;'>typing{dots}</span>",
                unsafe_allow_html=True,
            )
            time.sleep(0.01)

        # عرض الرد كلمة كلمة
        reply_so_far += reply_words[i] + " "
        bot_display.markdown(
            f"<b style='color: gray; font-family: Verdana, sans-serif;'>MentalEase:</b> "
            f"<span style='font-family: Georgia, serif; font-size: 16px;'>{reply_so_far.strip()}</span>",
            unsafe_allow_html=True,
        )

    typing_placeholder.empty()

    # بعد انتهاء الرد نضيفه للتاريخ
    chat_history.append({"role": "MentalEase", "content": full_reply})
    st.session_state.waiting_for_bot = False
    st.session_state.bot_response_stream = ""
    st.session_state.user_input_temp = ""
