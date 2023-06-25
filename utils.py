import os 
import toml
config = toml.load("config.toml")
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = config["PINECONE_API_KEY"]

import pinecone
pinecone.init(environment=config["PINECONE_API_ENV"])
PINECONE_INDEX_NAME = config["PINECONE_INDEX_NAME"]

from langchain.llms import OpenAI 
from langchain.vectorstores import Pinecone 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import re
import time
import requests
import pandas as pd 
import altair as alt
from lxml import html
import streamlit as st
from pathlib import Path
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text

__SAVE_DIR__ = Path("./fomc")
__SAVE_DIR__.mkdir(exist_ok=True, parents=True)

def download_pdf(url):
  resp = requests.get(url)
  resp.raise_for_status()

  fname = url.split("/")[-1]
  with open(__SAVE_DIR__/fname, "wb") as f:
    f.write(resp.content)

def download_fomc_statements():
  base_url = "https://www.federalreserve.gov"
  resp = requests.get(base_url + "/monetarypolicy/fomccalendars.htm")
  resp.raise_for_status()
  tree = html.fromstring(resp.text)
  html_links = tree.xpath("//a[starts-with(@href, '/newsevents/pressreleases/') and text()='HTML']")
  for link in html_links:
    url = base_url + link.get("href")
    print(url)

    resp = requests.get(url)
    resp.raise_for_status()
    tree = html.fromstring(resp.text)
    content = tree.find(".//div[@id='article']").text_content().strip()

    fname = url.split("/")[-1].replace(".htm", ".txt")
    with open(__SAVE_DIR__/fname, "w") as f:
      f.write(content)

    time.sleep(1)

@st.cache_data(ttl=3600, show_spinner=False)
def extract_fed_funds_rate(model="gpt-3.5-turbo", n=2):
  if n > 10:
    raise RuntimeError("Cannot parse more than 10 FOMC statements..")  

  fomc_statement = Object(
    id="fomc_statement",
    description="the FOMC statement",
    attributes=[
      Text(
        id="decision",
        description="FOMC decision on federal funds rate change",
      ),
      Text(
        id="rate_lower_bound",
        description="the federal funds rate (lower bound)",
      ),
      Text(
        id="rate_upper_bound",
        description="the federal funds rate (upper bound)",
      )
    ],
    examples=[
      (
        "the Committee decided to raise the target range for the federal funds rate to 4-3/4 to 5 percent",
        [{"decision": "up", "rate_lower_bound": "4.75%", "rate_upper_bound": "5%"}],
      ),
      (
        "the Committee decided to maintain the target range for the federal funds rate at 5 to 5-1/4 percent",
        [{"decision": "hold", "rate_lower_bound": "4.75%", "rate_upper_bound": "5%"}],
      ),
    ],
  )
  llm = ChatOpenAI(model=model, temperature=0, max_tokens=2000)
  chain = create_extraction_chain(llm, fomc_statement)

  out = []
  for fpath in sorted(__SAVE_DIR__.iterdir(), reverse=True)[:n]:

    date = re.search("(\d{8})", fpath.stem).group(1)
    date = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")

    with open(fpath) as f:
      txt = f.read().strip()
      txt = " ".join(txt.replace("Share","").split())

    output = chain.predict_and_parse(text=txt)["data"]["fomc_statement"][0]
    out.append([date, output])

  # formatting
  data = pd.DataFrame(out, columns=["Date", "Decision"])
  data["Date"] = pd.to_datetime(data["Date"])
  data = pd.concat([
    data.drop("Decision", axis=1),
    data["Decision"].apply(pd.Series),
  ], axis=1)
  data["decision"] = data["decision"].str.upper()
  
  str_to_float = lambda x: float(x.replace("%","")) / 100
  data[["rate_lower_bound", "rate_upper_bound"]] = data[["rate_lower_bound", "rate_upper_bound"]].applymap(str_to_float)

  data.rename(columns={"decision": "DECISION", "rate_lower_bound": "TARGET RATE (LOWER)", "rate_upper_bound": "TARGET RATE (UPPER)"}, inplace=True)
  return data

def format_streamlit_dataframe(data):
  # data = output from extract_fed_funds_rate
  styler = data.style.format({
    "Date": "{:%b %Y}",
    "TARGET RATE (LOWER)": "{:.2%}",
    "TARGET RATE (UPPER)": "{:.2%}",
  })
  return styler
  
def format_streamlit_barchart(data):
  # data = output from extract_fed_funds_rate
  bar_chart = alt.Chart(data).mark_bar().encode(
    x=alt.X("Date:O", timeUnit="yearmonth", axis=alt.Axis(format="%b %y")).title("Date"),
    y="TARGET RATE (LOWER)",
  ).configure(numberFormat=".2%")
  return bar_chart

@st.cache_data(ttl=3600, show_spinner=False)
def get_completion_QA(user_input, model="text-davinci-003"):
  loader = DirectoryLoader(path="./fomc", glob="./monetary2023*.txt", loader_cls=TextLoader)
  doc_texts = loader.load()
  for doc in doc_texts:
    doc.page_content = " ".join(doc.page_content.replace("Share","").split())

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
  texts = text_splitter.split_documents(doc_texts)

  embeddings = OpenAIEmbeddings()
  doc_search = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=PINECONE_INDEX_NAME)

  llm = OpenAI(model=model, temperature=0, max_tokens=500)
  chain = load_qa_chain(llm, chain_type="stuff")

  query = f"""
  You are a financial analyst whose job is to analyze FOMC statements and how the federal fund rates changed over time. \
  You will be given a user question below, which follows <<< USER QUESTION >>>. \
  Respond to the user question in a friendly and helpful tone, with very concise answers. \
  If the user is asking about specific FOMC meeting, make sure the meeting occurred by checking the FOMC meeting date. \ 
  The FOMC meeting date can be found at the bottom of each document, in the following format: Implementation Note issued 'FOMC MEETING DATE' e.g. Implementation Note issued June 14, 2023
  If the meeting did not occur, please say the meeting didn't occur in that month.

  <<< USER QUESTION >>>
  {user_input} \
  """

  docs = doc_search.similarity_search(query)
  response = chain.run(input_documents=docs, question=query)
  return response