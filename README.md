# LLMatchup
LLMatchup - Two LLMs face off to see which one is faster, better, stronger. Or at least faster and more precise.

You will need Ollama and at least two of their models installed. The LLMs will then be put to the test, answering questions (found in questions/questions.csv) based on the given context (in my case, Homer's "The Odyssey" in PDF format). A third, impartial LLM (it can be one of the models in the competition - they are all separate instances) will determine if the contestant's response is correct, based on the answer given in the questions.csv file.

# Setup
1. Download and install Ollama.

2. Download and install Python.

3. Copy the project over to C:\LLMatchup

4. Open a Command Prompt as Administrator.

5. cd \LLMatchup

6. In the Ollama website, find two or more models and pull them into ollama:
   ollama pull <LLM_1>
   ollama pull <LLM_2>
   I had better results with the larger (8b+) models. The 1.x models were very spotty.

8. pip install ollama langchain langchain_community langchain_huggingface langchain_ollama pdfplumber faiss-cpu matplotlib

9. python app.py
