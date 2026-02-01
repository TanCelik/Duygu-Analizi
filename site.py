import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Duygu Analizörü", page_icon="🤖")

@st.cache_resource
def model_yukle():
    # Bu satır gerçek bir derin öğrenme modelini çağırır
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("🤖 Gerçek AI Duygu Analizi")
st.write("Bu uygulama arkada Tan'ın devasa taşaklarını çalıştırıyor.")

user_input = st.text_input("İngilizce bir cümle girin:")

if user_input:
    classifier = model_yukle()
    res = classifier(user_input)[0]
    
    label = res['label']
    score = res['score']
    
    if label == "POSITIVE":
        st.success(f"Sonuç: POZİTİF (Güven: %{score*100:.2f})")
    else:
        st.error(f"Sonuç: NEGATİF (Güven: %{score*100:.2f})")
