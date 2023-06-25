# FedGPT
A streamlit app that analyzes FOMC statements using OpenAI, LangChain, Pinecone, and [Kor](https://eyurtsev.github.io/kor/).

## Demo
<p align="center">
  <img src="demo.gif" width="500" />
</p>

## How to use
Before you can run the app, you need to update the config.toml file with your OpenAI and Pinecone API keys. 
After installing the packages required using `requirements.txt` file, you can run the following command to launch the streamlit app.
```python
streamlit run app.py
```

There are 2 main features in the app:

1. Q&A where you can ask GPT any question you want on past FOMC statements.
2. Summary which shows the historical rate decisions in a bar chart.
