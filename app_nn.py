# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set Thai font
#mpl.font_manager.fontManager.addfont('thsarabunnew-webfont.ttf')
#mpl.rc('font', family='TH Sarabun New', size=12)

# โหลด stopwords ภาษาไทย
thai_stopwords_set = set(thai_stopwords())

# ฟังก์ชันทำความสะอาดข้อความ
def remove_emojis_and_symbols(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s]", "", text)
    return text

def clean_text_combined(text):
    if isinstance(text, str):
        text = remove_emojis_and_symbols(text)
        tokens = word_tokenize(text, engine='newmm')
        seen = set()
        unique_tokens = [token for token in tokens if not (token in seen or seen.add(token))]
        filtered_tokens = [token for token in unique_tokens if token not in thai_stopwords_set and token.strip()]
        return " ".join(filtered_tokens)
    return text

# Load model & tools
@st.cache_resource
def load_all_models():
    model = joblib.load('final_model_NN.pkl')
    vectorizer = joblib.load('vectorizer_NN.pkl')
    encoder = joblib.load('label_encoder_NN.pkl')
    selector = joblib.load('selector_NN.pkl')
    return model, vectorizer, encoder, selector

model, vectorizer, encoder, selector = load_all_models()

# Title
st.title("🔍 ระบบจำแนกข้อความด้วย Neural Network")

# File uploader
uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ที่มีข้อความ (คอลัมน์แรกเป็นข้อความ)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("โหลดไฟล์สำเร็จ!")
    
    st.write("📌 ตัวอย่างข้อมูล:")
    st.dataframe(df.head())

    cleaned_new_texts = [clean_text_combined(text) for text in df.iloc[:,0]]
    vect_new_texts = vectorizer.transform(cleaned_new_texts)
    selected_features_new_texts = selector.transform(vect_new_texts)  # Apply selector
    new_predictions_probs = model.predict(selected_features_new_texts)
    new_predictions_classes = np.argmax(new_predictions_probs, axis=1)
    decoded_predictions = encoder.inverse_transform(new_predictions_classes)

    # แสดงผลลัพธ์
    results_df = pd.DataFrame({
        'ข้อความต้นฉบับ': df.iloc[:,0],
        'หมวดหมู่ที่ทำนายได้': decoded_predictions
    })

    st.write("✅ ผลการทำนาย:")
    st.dataframe(results_df)

    # ดาวน์โหลด
    csv_output = results_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 ดาวน์โหลดผลลัพธ์เป็น CSV", data=csv_output, file_name="prediction_results.csv", mime="text/csv")
else:
    st.info("กรุณาอัปโหลดไฟล์ CSV ก่อนเริ่มการประมวลผล")
