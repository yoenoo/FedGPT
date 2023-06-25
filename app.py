import time
import random
import streamlit as st
from utils import extract_fed_funds_rate, format_streamlit_dataframe, format_streamlit_barchart, get_completion_QA

MODEL_QA = "text-davinci-003"
MODEL_PARSER = "gpt-3.5-turbo"

st.title("FOMC Statement Analyzer")

option = st.radio("Select option", ("Q&A", "Summary"), horizontal=True)
if option == "Q&A":
  random_prompt = st.button("Generate a random question!", type="primary", use_container_width=True)
  text_area = st.empty()
  text_area_label = "What is the question you have in mind?"
  user_input = text_area.text_area(label=text_area_label, value="")

  if random_prompt:
    example_prompts = [
      "When did the most recent FOMC meeting occur?",
      "What was the FOMC committee's decision on rate hikes on the June meeting?",
      "What was the reason behind the FOMC committee's decision to raise interest rate on the most recent FOMC meeting?",
      "Is Jerome Powell worried about the banking crisis?",
    ]
    user_input_selected = random.choice(example_prompts)
    user_input = text_area.text_area(label=text_area_label, value=user_input_selected)
    time.sleep(1)

  if user_input:
    response = get_completion_QA(user_input, model=MODEL_QA)
    st.markdown("""---""")
    st.write(response)
elif option == "Summary":
  docs_to_parse = 5
  progress_text = f"Please wait for {MODEL_PARSER} to parse FOMC statements..."  
  progress_bar = st.progress(0.0, text=progress_text)
  for i in range(docs_to_parse):
    data = extract_fed_funds_rate(model=MODEL_PARSER, n=docs_to_parse)
    progress = (i+1) / float(docs_to_parse)
    progress_bar.progress(progress, text=progress_text)

  time.sleep(0.5)
  progress_bar.empty()

  styler = format_streamlit_dataframe(data)
  st.dataframe(styler, hide_index=True, use_container_width=True)

  bar_chart = format_streamlit_barchart(data)
  st.altair_chart(bar_chart, use_container_width=True)
else:
  raise RuntimeError(f"Please select valid option: {option}")