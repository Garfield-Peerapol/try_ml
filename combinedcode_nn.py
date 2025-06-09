import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import matplotlib as mpl
import matplotlib.pyplot as plt

# ตั้งค่า font ภาษาไทย
mpl.font_manager.fontManager.addfont('thsarabunnew-webfont.ttf')
mpl.rc('font', family='TH Sarabun New', size=12)

thai_stopwords_set = set(thai_stopwords())

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

@st.cache_resource
def load_models():
    model = joblib.load('final_model_NN.pkl')
    vectorizer = joblib.load('vectorizer_NN.pkl')
    encoder = joblib.load('label_encoder_NN.pkl')
    selector = joblib.load('selector_NN.pkl')
    return model, vectorizer, encoder, selector

st.title("🧠 ระบบจำแนกข้อความด้วย Neural Network")

uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ที่มีข้อความ (คอลัมน์แรกเป็นข้อความ)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("โหลดไฟล์สำเร็จ!")
    st.write("ตัวอย่างข้อมูล:")
    st.dataframe(df.head())

    model, vectorizer, encoder, selector = load_models()

    cleaned_texts = [clean_text_combined(text) for text in df.iloc[:, 0]]
    vect_texts = vectorizer.transform(cleaned_texts)
    selected_features = selector.transform(vect_texts)

    # ทำนายผล
    predictions_probs = model.predict(selected_features)
    predictions_classes = np.argmax(predictions_probs, axis=1)
    decoded_predictions = encoder.inverse_transform(predictions_classes)

    results_df = pd.DataFrame({
        'ข้อความต้นฉบับ': df.iloc[:, 0],
        'หมวดหมู่ที่ทำนายได้': decoded_predictions
    })

    st.write("ผลการทำนาย:")
    st.dataframe(results_df)

    csv_output = results_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("ดาวน์โหลดผลลัพธ์เป็น CSV", data=csv_output, file_name="prediction_results.csv", mime="text/csv")
else:
    st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มการทำนาย")
