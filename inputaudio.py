import streamlit as st
from openai import AzureOpenAI
import tempfile
import os
import json

#config whisper as transcription
client = AzureOpenAI(
    api_key="id",
    api_version="date",
    azure_endpoint="endpoint model"
)

MAX_CONTENT_SIZE = 25 * 1024 * 1024

#audio transcription
def transcribe_audio(file_path):
    # Prompt transcribe
    response = client.audio.transcriptions.create(
        model="whisper-playground",
        file=open(file_path, "rb") #upload audio file
    )
    transcription = response.text

    return transcription

#config gpt 4 for sentiment analysis
def CustomChatGPT(user_input):
    client = AzureOpenAI(
        api_key="id",
        api_version="date",
        azure_endpoint="endpoint model"
    )
    #Prompt summarize transcription
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                You are an AI assistant that evaluate conversation between customer and operator.
    
                please evaluate these point by customer :
                1. Introduction of operator
                2. operator ask receiver's consent to continue conversation
                3. answer of consent
                4. Operator info call has being recorded
                5. Operator Sentiment
                6. Receiver sentiment
                7. Call disconnected before conversation ends
    
                please create a flag based on each point above and summarize it into a table
                """,
            },
            {
                "role": "assistant",
                "content": """please format as json with keys  
                Introduction of operator, operator ask receiver's consent to continue conversation, answer of consent, Operator info call has being recorded, Operator Sentiment, Receiver sentiment, Call disconnected before conversation ends
                """
            },
            {
                "role": "user",
                "content": f"berikut adalah percakapan yang akan di evaluasi antara operator dan customer: \n {user_input}"
            }
        ],
        temperature=0
    )
    message = completion.choices[0].message.content
    return message

st.title('Title Your Web App')

# File uploader
uploaded_file = st.file_uploader("Upload an audio file")

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    st.audio(uploaded_file, format='audio/mp3')

    # Transcribe the audio
    with st.spinner('Transcribing audio...'):
        transcription = transcribe_audio(temp_file_path)
        st.success('Transcription complete!')

    st.subheader('Transcription')
    st.text_area('Transcription', transcription, height=300)

    # Summarize the transcription
    with st.spinner('Summarizing text...'):
        summary = CustomChatGPT(transcription)
        st.success('Summarization complete!')

    st.subheader('Summary')
    st.text_area('Summary', summary, height=200)

    # Clean up temporary file
    os.remove(temp_file_path)
