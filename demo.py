import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("FirstAidMate: Chatbot using NLP")

        # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize chat history
    if "counter" not in st.session_state:
        st.session_state.counter = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

        # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{st.session_state.counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

         # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to develop an AI-powered first-aid chatbot that can provide immediate, accurate, and life-saving first-aid guidance based on user input. The chatbot is built using Natural Language Processing (NLP) techniques and a machine learning model (Logistic Regression) to understand emergency scenarios and recommend appropriate first-aid steps.\n This chatbot is designed to assist users in real-time emergency situations by providing quick, reliable, and structured first-aid information. It covers a wide range of medical emergencies, from CPR and choking to burns, fractures, and allergic reactions.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two major parts:

        NLP-Based First-Aid Intent Classification:

        The chatbot is trained using Logistic Regression on a dataset containing labeled intents and first-aid conditions.
        It analyzes the user's query to identify medical emergencies, symptoms, and recommended first-aid procedures.
        The chatbot extracts entities like symptoms, conditions, and injuries to provide context-aware responses.
        Interactive Chatbot Interface using Streamlit:

        A user-friendly chatbot interface is built using Streamlit, a Python framework for web-based applications.
        The interface allows users to input first-aid-related questions and receive instant responses based on the trained model.
        The chatbot provides step-by-step first-aid instructions, making it useful in real-life emergency situations.
        """)

        st.subheader("Dataset:")

        st.write("""
        The chatbot's responses are based on a structured dataset of first-aid information, designed to handle a variety of emergency conditions.
                 \n

        Intents: Represents different medical emergencies (e.g., "CPR", "Heart Attack", "Fracture", "Snake Bite").
                 \n
        Entities: Extracts key details from user queries (e.g., "chest pain", "unconscious person", "burn treatment").
                 \n
        User Input Processing: The chatbot understands natural language queries and matches them to appropriate first-aid responses.
                 \n
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("""
        The chatbot interface is built with Streamlit, ensuring an interactive, real-time response system.\n

         ✅User-Friendly Input: Users can type their queries and receive structured first-aid instructions immediately.\n
         ✅Context-Aware Responses: The chatbot analyzes symptoms before suggesting first-aid steps.\n
         ✅Emergency Assistance: Provides urgent guidance for life-threatening conditions while advising users to call emergency services (112   in India) when needed.\n
        """)
        
        st.subheader("Conclusion:")

        st.write("""This first-aid chatbot serves as an AI-powered emergency assistant, helping users handle medical emergencies effectively. The chatbot leverages NLP and machine learning to understand and classify user queries while providing accurate first-aid guidance.

The project can be further improved by:\n
✅ Expanding the dataset to include more medical emergencies.\n
✅ Enhancing NLP capabilities with deep learning models.\n
✅ Integrating voice commands for hands-free assistance.\n

This chatbot has the potential to save lives by providing instant first-aid guidance in emergency situations. """)

if __name__ == '__main__':
    main()