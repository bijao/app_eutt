import streamlit as st


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
 
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
 
 
st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        predicted_class = model.config.id2label[predicted_class_id]
        st.write('Sentiment: ')
        if predicted_class == 'POSITIVE':
            st.markdown("<h1 style='text-align: center; color: green;'>&#128512; Je suis content</h1>", unsafe_allow_html=True)
        elif predicted_class == 'NEGATIVE':
            st.markdown("<h1 style='text-align: center; color: red;'>&#128544; Je ne suis pas content</h1>", unsafe_allow_html=True)     

