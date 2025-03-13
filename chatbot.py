# import os
# import json
# import datetime
# import csv
# import random
# import nltk
# import ssl
# import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# # Handling SSL for NLTK
# ssl._create_default_https_context = ssl._create_unverified_context
# nltk.data.path.append(os.path.abspath("nltk_data"))
# nltk.download('punkt')

# # Load intents from JSON
# file_path = os.path.abspath("intents.json")
# with open(file_path, "r") as file:
#     intents = json.load(file)

# # Setup model
# vectorizer = TfidfVectorizer()
# clf = LogisticRegression(random_state=0, max_iter=10000)

# # Prepare Data
# tags = []
# patterns = []
# for intent in intents:
#     for pattern in intent['patterns']:
#         tags.append(intent['tag'])
#         patterns.append(pattern)

# # Train Model
# x = vectorizer.fit_transform(patterns)
# y = tags
# clf.fit(x, y)

# # Chatbot Function
# def chatbot(input_text):
#     input_text = vectorizer.transform([input_text])
#     tag = clf.predict(input_text)[0]
#     for intent in intents:
#         if intent['tag'] == tag:
#             return random.choice(intent['responses'])
#     return "I'm not sure how to help. Try asking differently."

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="FirstAidMate", page_icon="🚑", layout="wide")

#     st.markdown("""
#         <style>
#             .stTextInput>div>div>input { font-size: 16px !important; }
#             .chat-container { 
#                 padding: 10px; 
#                 background: #f7f9fc; 
#                 border-radius: 10px; 
#                 margin-bottom: 10px;
#             }
#             .chatbot-response {
#                 color: black;
#                 padding: 10px;
#                 background-color: #d9edf7;
#                 border-radius: 10px;
#                 font-weight: bold;
#             }
#             .user-input {
#                 color: black;
#                 padding: 10px;
#                 background-color: #dff0d8;
#                 border-radius: 10px;
#                 font-weight: bold;
#             }
#         </style>
#     """, unsafe_allow_html=True)

#     st.title("🚑 FirstAidMate: AI-Powered First Aid Chatbot")

#     # Sidebar Menu
#     menu = ["💬 Chat", "📜 History", "ℹ️ About"]
#     choice = st.sidebar.radio("Menu", menu)

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     if choice == "💬 Chat":
#         st.subheader("💡 Ask First-Aid Questions")
#         st.write("👋 **Type your query below and press enter**")

#         user_input = st.text_input("You:", key="user_input")
#         if user_input:
#             response = chatbot(user_input)

#             # Save chat history
#             st.session_state.chat_history.append(("You", user_input))
#             st.session_state.chat_history.append(("Chatbot", response))

#             # Save to CSV
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
#                 csv_writer = csv.writer(csvfile)
#                 csv_writer.writerow([user_input, response, timestamp])

#     # Display Chat History with Chat Bubbles
#     if st.session_state.chat_history:
#         st.markdown("### 🗨️ Chat History")
#         for role, text in st.session_state.chat_history:
#             if role == "You":
#                 st.markdown(f"<div class='chat-container user-input'>🧑‍💻 <b>{role}:</b> {text}</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<div class='chat-container chatbot-response'>🤖 <b>{role}:</b> {text}</div>", unsafe_allow_html=True)

#     elif choice == "📜 History":
#         st.subheader("📜 Conversation History")
#         if os.path.exists('chat_log.csv'):
#             with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
#                 csv_reader = csv.reader(csvfile)
#                 next(csv_reader)  # Skip header
#                 for row in csv_reader:
#                     st.markdown(f"🧑‍💻 **User:** {row[0]}")
#                     st.markdown(f"🤖 **Chatbot:** {row[1]}")
#                     st.markdown(f"🕒 **Timestamp:** {row[2]}")
#                     st.markdown("---")
#         else:
#             st.write("No history found.")

#     elif choice == "ℹ️ About":
#         st.subheader("🚀 About FirstAidMate")
#         st.write("""
#         FirstAidMate is an AI chatbot designed to provide **immediate** first-aid guidance. It helps users in emergencies by giving **step-by-step** instructions.
        
#         💡 **Features:**  
#         - Instant responses to **first-aid queries**  
#         - Covers multiple emergencies like CPR, burns, fractures, choking, etc.  
#         - Uses **NLP & Machine Learning** for smart responses  
#         - **Logs conversations** for future improvements  
#         - Interactive **chat UI with bubbles & icons**  
#         - Quick **emergency contact information**  
#         """)

#         st.subheader("🛠️ Technologies Used")
#         st.write("""
#         - **Python** (Machine Learning, NLP)  
#         - **Scikit-Learn** (Logistic Regression Model)  
#         - **NLTK** (Natural Language Processing)  
#         - **Streamlit** (Interactive Web App)  
#         - **JSON & CSV** (Data Storage)  
#         """)

# if __name__ == '__main__':
#     main()

import os
import json
import datetime
import csv
import random
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Handling SSL for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Setup model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare Data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train Model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot Function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to help. Try asking differently."

# Streamlit UI
def main():
    st.set_page_config(page_title="FirstAidMate", page_icon="🚑", layout="wide")

    st.markdown("""
        <style>
            .stTextInput>div>div>input { font-size: 16px !important; }
            .chat-container { 
                padding: 10px; 
                background: #f7f9fc; 
                border-radius: 10px; 
                margin-bottom: 10px;
            }
            .chatbot-response {
                color: black;
                padding: 10px;
                background-color: #d9edf7;
                border-radius: 10px;
                font-weight: bold;
            }
            .user-input {
                color: black;
                padding: 10px;
                background-color: #dff0d8;
                border-radius: 10px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚑 FirstAidMate: AI-Powered First Aid Chatbot")

    # Sidebar Menu
    menu = ["💬 Chat", "🗄 History", "ℹ️ About"]
    choice = st.sidebar.radio("Menu", menu)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if choice == "💬 Chat":
        st.subheader("💡 Ask First-Aid Questions")
        st.write("👋 **Type your query below and press enter**")

        user_input = st.text_input("You:", key="user_input")
        if user_input:
            response = chatbot(user_input)

            # Save chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Chatbot", response))

            # Save to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Display Chat History with Chat Bubbles
            if st.session_state.chat_history:
                st.markdown("### 👨‍🎓 Chat History")
                for role, text in st.session_state.chat_history:
                    if role == "You":
                        st.markdown(f"<div class='chat-container user-input'>🤖 <b>{role}:</b> {text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-container chatbot-response'>🤖 <b>{role}:</b> {text}</div>", unsafe_allow_html=True)

    elif choice == "🗄 History":
        st.subheader("🗄 Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.markdown(f"👨‍🎓 **User:** {row[0]}")
                    st.markdown(f"🤖 **Chatbot:** {row[1]}")
                    st.markdown(f"🕛 **Timestamp:** {row[2]}")
                    st.markdown("---")
        else:
            st.write("No history found.")


    elif choice == "ℹ️ About":
        st.subheader("🚀 About FirstAidMate")
        st.write("""
        FirstAidMate is an AI chatbot designed to provide **immediate** first-aid guidance. It helps users in emergencies by giving **step-by-step** instructions.
        
        💡 **Features:**  
        - Instant responses to **first-aid queries**  
        - Covers multiple emergencies like CPR, burns, fractures, choking, etc.  
        - Uses **NLP & Machine Learning** for smart responses  
        - **Logs conversations** for future improvements  
        - Interactive **chat UI with bubbles & icons**  
        - Quick **emergency contact information**  
        """)

        st.subheader("🛠️ Technologies Used")
        st.write("""
        - **Python** (Machine Learning, NLP)  
        - **Scikit-Learn** (Logistic Regression Model)  
        - **NLTK** (Natural Language Processing)  
        - **Streamlit** (Interactive Web App)  
        - **JSON & CSV** (Data Storage)  
        """)

if __name__ == '__main__':
    main()
