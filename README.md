
## 📰 Fake or Fact

A predictive analysis web application using NLP and Deep Learning to classify news articles as Fake or Real. The model is built with LSTM architecture, enabling accurate text-based predictions. Developed with Python, Streamlit, and managed using DVC and DagsHub for seamless data and model version control.

## 🧠 Project Overview
Fake or Fact is a machine learning project designed to identify the authenticity of news articles using advanced Natural Language Processing (NLP) and Deep Learning techniques. This solution leverages an LSTM (Long Short-Term Memory) neural network to effectively understand the context and semantics of textual data for accurate classification.
![fake_news](https://github.com/gowthamnagu/fakenewsclassifier/blob/main/images/fake_news.png)
![real_news](https://github.com/gowthamnagu/fakenewsclassifier/blob/main/images/Real_news.png)
![real_news](https://github.com/gowthamnagu/fakenewsclassifier/blob/main/images/mlflow.png)

## 🛠️ Tech Stack
- **Python 3.11**
- **LSTM (Keras, TensorFlow)** — Deep learning model for text classification
- **Streamlit** — Web UI for interactive model prediction
- **DVC** — Data and model version control
- **DagsHub** — Remote repository for code, data, and experiment tracking
- **Pandas, NumPy, Scikit-learn, Nltk** — Data preprocessing and utilities
- **NLP Techniques** —Tokenization,Stopwords removal, Stemming, etc.
## 🚀 Installation
    

- **Clone the repository**
    
       git clone https://dagshub.com/gowthambreeze/fakeorfact.git

- **Set up a virtual environment** 

        conda create -p venv python=3.11 -y
        conda activate venv/

- **Install dependencies** 

        pip install -r requirements.txt

- **Pull data and models using DVC** 

        dvc pull        
## ▶️ Running the Application

Run the main file to start the application:

    streamlit run app.py
## 🗂️ Project Structure
```
.
├── data/
│   ├── processed/
│   │   └── data.csv
│   └── raw/
│       ├── data.csv
│       └── data.csv.dvc
├── models/
│   └── model.keras
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── train.py
├── dvc.yaml
├── params.yaml
├── app.py
└── README.md
```
## 📊 Data Pipeline

![Data Pipeline](https://github.com/gowthamnagu/fakenewsclassifier/blob/main/images/datapipeline.png)

## 🧪 Experiment Tracking with DagsHub

All code, data, models, and metrics are versioned using DVC and tracked on DagsHub and MLFlow.

DagsHub Repo: https://dagshub.com/gowthambreeze/fakenewsclassifier.git

## 📄 License

This project is open-source under the MIT License.


## 🙋‍♂️ Contributions

Feel free to fork the repository, make improvements, and create pull requests!
---
Let me know if you want a downloadable PDF version of this `README.md` or help linking your GitHub repo to GitHub Pages for a live project site.

