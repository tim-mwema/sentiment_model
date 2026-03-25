import os
os.environ["HF_HOME"] = "F:/hf_cache"  # Changed to F: to any drive with >1GB free
os.makedirs("F:/hf_cache", exist_ok=True)
#import all modules 
import streamlit as st 
import pandas as pd 
from transformers import pipeline
import torch
import accelerate

#page configuration for streamlit app
st.set_page_config(
    page_title="Sentiment Analysis HR Tool ",
    layout='centered'
)
#title and description of  the application
st.title("HR AI sentiment Analysis Application" )
st.write("Welcome to the HR Sentiment Analysis AI! This app uses a **Large Language Model** to detect the sentiment of employee reviews")
# load the model(cache it for performance enhancement) 
@st.cache_resource
def load_sentiment_model():
    #DistilBERT  model
    model=pipeline("text-classification",model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    return model
#load the model and show the spinner while sentiment is being generated
with st.spinner("Loading the HR Model...please wait...."):
    sentiment_pipeline = load_sentiment_model ()
    
#main interface of the application
tab1,tab2 = st.tabs(["Analyse typed text","Analyse uploaded CSV Dataset"])
#tab1
with tab1:
    st.header("Analyze the single review ")
    user_input = st.text_area("Please enter the employee review","The organization does not have medical cover")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            #run the model
            result=sentiment_pipeline(user_input)
            ##Extracting the result(sentiment), and the score
            label = result[0] ["label"]
            score = result[0] ["score"]
            #display the results
            st.success(f"Sentiment:**{label}**")
            st.info(f"Confidence score on the above sentiment: **{score:.2f}**")
        else:
            st.warning("Please enter some text value to analyze the sentiment")
#tabs2
with tab2:
    st.header("Analyse the Dataset(csv)")
    uploaded_file = st.file_uploader("upload your csv with employee sentiment")
    
    if uploaded_file is not None:
        #reading the csv file
        df=pd.read_csv(uploaded_file)
        #display first few rows of the file 
        st.write("###Preview of the dataset:")
        st.dataframe(df.head())
        
        # check if 'review_text' column in the dataset uploaded
        text_column = "review_text"
        
        if text_column in df.columns:
            if st.button("Analyze the entire dataset"):
                st.write("Analysing...this might take a while")
                #apply the model to the text column to analyse the sentiment
                df["AI_Sentiment"] = df[text_column].apply(lambda x: sentiment_pipeline(str(x)[:400], truncation=True, max_length=512)[0]["label"])
                
                #display the results
                st.write("### Analysis Completed!")
                st.dataframe(df[[text_column,'AI_Sentiment']].head(20))
                
                # show a simple bar chart to display the sentiment
                st.write("### Sentiment Distribution")
                sentiment_counts = df ['AI_Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
                
                #download the result
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download the sentiment data",
                    data=csv,
                    file_name="Sentiment Analysis file.csv",
                    mime="text/csv"
                )
                
        else:
            st.error(f"Could not find the column {text_column} in your csv file.Please check for the column names and rename the column")
