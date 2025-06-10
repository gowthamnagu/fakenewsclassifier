import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import yaml
import os

nltk.download("stopwords")
#load parameter from params.yaml file
params=yaml.safe_load(open("params.yaml"))["preprocess"]
voc_size=5000
sent_length=20
def preprocess(input_path, output_path):
    # Load data
    data = pd.read_csv(input_path)
    
    # Initialize tools
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    corpus = []
    for i in range(len(data)):
        text = str(data.loc[i, 'title']) if pd.notnull(data.loc[i, 'title']) else ""
        # Clean text: remove non-alphabet chars, lowercase, tokenize
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        
        # Remove stopwords and stem
        review = [ps.stem(word) for word in review if word not in stop_words]
        review = ' '.join(review)
        corpus.append(review)
        
    
    # Create processed DataFrame with text and label
    onehot_repr=[one_hot(words,voc_size)for words in corpus] 
    embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length)
    processed_df = pd.DataFrame({
        'title': embedded_docs.tolist(),
        'label': data['label']
    })
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data
    processed_df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__=="__main__":
    preprocess(params["input"],params["output"])    