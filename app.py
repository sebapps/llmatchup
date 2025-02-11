# LLMatchup
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from datetime import datetime
from ollama import ListResponse, list
import os
import time
import csv
import matplotlib.pyplot as plt

# Path to PDF to index
path_to_pdf = "pdfs/the-odyssey.pdf"

# Path to the questions
path_to_questions = "questions/questions.csv"

# RAG class to encapsulate different models
class RAGClass:
    def __init__(self, model_name: str = "llama2:7b-chat-q4"):
        
        # Load the language model (LLM)
        self.llm = OllamaLLM(model=model_name)
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context. Be concise.
        If you cannot find the answer in the context, say "I cannot answer this based on the provided context."
        The question should be answered in one word if possible.  If not, then a couple of words, at the most. 
        Be very concise and precise. Do not display your thought process. Please simply answer the question with
        simply one word, if possible.
                                                       
        Context: {context}
        Question: {question}
        Answer: """)

    # Initialize the RAG chain to perform the retrieval from the vector store (FAISS index)
    def setup_rag_chain(self, vectorstore: FAISS):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2, "fetch_k": 3})
            
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    # Invoke the question to the LLM
    def query(self, chain, question: str) -> str:
        return chain.invoke(question)

# Class to encapsulate the referee model
class RefereeClass:
    def __init__(self, model_name: str = "llama2:7b-chat-q4"):
        
        # Load the language model (LLM)
        self.llm = OllamaLLM(model=model_name)
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        You are given a student's response and the correct answer. Your job is to verify that the response matches the answer. If the response matches the answer,
        you will output the word "True". If the response does not match the answer, you will output the word "False". You will only output one
        of those two responses. The response has to match the answer in order for it to be correct. You will use your judgment for correctness. 
        An example would be if the response is 7 and the answer is Seven, then you will output "True" since the answer is Seven and the number 
        written out is also Seven. Limit your response to one word and two tokens.

        Context: The response is: {response}. The answer is: {answer}.                         
        Question: The response is: {response}. The answer is: {answer}. Please output "True" if the response matches the answer or "False" otherwise.
        Answer: """)

    # Invoke the question to the LLM
    def query(self, response: str, answer: str) -> str:
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"response" : response, "answer": answer})

# Part of the embedding process- create a vector store for the chunks, and save it as a FAISS index,
# so that it does not have to be created more than once.
def create_vectorstore() -> FAISS:

    # Initialize embeddings using a lightweight model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    if os.path.exists("faiss_index"):
        # FAISS index already exists - simply load it
        print("Vector store already created. Loading...")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    else:
        # Create the vector store
        print("Creating the vector store...")
        loader = PDFPlumberLoader(path_to_pdf)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )

        splits = text_splitter.split_documents(documents)
        batch_size = 32
        vectorstore = FAISS.from_documents(splits[:batch_size], embeddings)

        for i in range(batch_size, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            vectorstore.add_documents(batch)
            vectorstore.save_local("faiss_index")    

    return vectorstore

# Main program
def main():

    # We need at least 2 LLMs installed through Ollama
    response: ListResponse = list()

    if len(response.models) < 2:
        print("Please install at least 2 LLMs to compare.")
        exit()

    rag1_name = ""
    rag2_name = ""
    referee_name = ""
    rag1 = ""
    rag2 = ""
    referee = ""

    # Select the two LLMs to compare
    if len(response.models) == 2:
        rag1_name = response.models[0].model
        rag2_name = response.models[1].model
        
    else:
        index = 0
        for model in response.models:
            print (str(index) + " - " + model.model)
            index = index + 1
        
        print(" ")
        index1 = -1
        index2 = -1
        while index1 == -1:
            try:
                index1 = int(input("Enter the number that corresponds to the first LLM to test.\n"))
                if index1 < 0 or index1 > (len(response.models) - 1):
                    print("Invalid input. Please try again")
            except ValueError:
                print("Invalid input. Please try again")
            
        while index2 == -1:
            try:
                index2 = int(input("Enter the number that corresponds to the second LLM to test.\n"))
                if index2 < 0 or index2 > (len(response.models) - 1):
                    print("Invalid input. Please try again")
            except ValueError:
                print("Invalid input. Please try again")
            
        rag1_name = response.models[index1].model
        rag2_name = response.models[index2].model

    # Select the referee
    if len(response.models) == 2:
        index = 0
        for model in response.models:
            print (str(index) + " - " + model.model)
            index = index + 1
        
        print(" ")
    
    referee_index = -1
    while referee_index == -1:
        try:
            referee_index = int(input("Enter the number that corresponds to the referee LLM.\n"))
            if referee_index < 0 or referee_index > (len(response.models) - 1):
                print("Invalid input. Please try again")
        except ValueError:
            print("Invalid input. Please try again")

    referee_name = response.models[referee_index].model

    # Create the actual RAG objects  
    rag1 = RAGClass(model_name=rag1_name)
    rag2 = RAGClass(model_name=rag2_name)
    referee = RefereeClass(model_name=referee_name)

    # Create and load the vector store
    vectorstore = create_vectorstore()

    # Set up the LangChain objects
    chain1 = rag1.setup_rag_chain(vectorstore)
    chain2 = rag2.setup_rag_chain(vectorstore)

    print("LLMs loaded. Beginning the question / answer portion. Please wait a few minutes.")

    # Load the CSV with the questions and answers
    index = 0
    time1 = 0.0
    time2 = 0.0
    correct1 = 0
    correct2 = 0

    time1_array = []
    time2_array = []
    correct1_array = []
    correct2_array = []

    with open(path_to_questions, "r") as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            if index > 0:
                question = row[0]
                print("\nProcessing question " + str(index) + "...")
                answer = row[1]

                # Ask the two LLMs the question
                start = time.time()
                response1 = rag1.query(chain1, question)
                end = time.time()
                time1_array.append(end-start)
                time1 = time1 + (end-start)

                # Strip the <think> tags
                if "</think>" in response1:
                    response1 = response1.split("</think>",1)[1].strip()
                
                # Ask the referee if the answer is correct
                referee_response = referee.query(response1, answer)
                
                # Strip the <think> tags
                if "</think>" in referee_response:
                    referee_response = referee_response.split("</think>",1)[1].strip()
                
                correct_answer = "true" in referee_response.lower().strip()
                if correct_answer:
                    correct1 = correct1 + 1
                    correct1_array.append("Y")
                else:
                    correct1_array.append("N")

                print("Question: " + question)
                print("Valid answer: " + answer)
                
                print("\n" + rag1_name + " Response Time: {:0.2f} seconds".format(end-start))
                print(rag1_name + " Response: " + response1)
                print("Referee Decision: " + referee_response.strip())
                print(rag1_name + " Correct response? %s" % correct_answer)

                start = time.time()
                response2 = rag1.query(chain2, question)
                end = time.time()
                time2_array.append(end-start)
                time2 = time2 + (end-start)

                # Strip the <think> tags
                if "</think>" in response2:
                    response2 = response2.split("</think>",1)[1].strip()
                
                # Ask the referee if the answer is correct
                referee_response = referee.query(response2, answer)

                # Strip the <think> tags
                if "</think>" in referee_response:
                    referee_response = referee_response.split("</think>",1)[1].strip()

                correct_answer = "true" in referee_response.lower().strip()
                if correct_answer:
                    correct2 = correct2 + 1
                    correct2_array.append("Y")
                else:
                    correct2_array.append("N")

                print("\n" + rag2_name + " Response Time: {:0.2f} seconds".format(end-start))
                print(rag2_name + " Response: " + response2)
                print("Referee Decision: " + referee_response)
                print(rag2_name + " Correct response? %s" % correct_answer)
            
            index = index + 1
    
    # Results
    print("\n\nFinal Results")
    print(rag1_name)
    print("Average response time: {:0.2f} seconds.".format(time1 / (index - 1)))
    print("Correct answers: " + str(correct1) + " out of " + str(index - 1) + ".\n")
    print(rag2_name)
    print("Average response time: {:0.2f} seconds.".format(time2 / (index - 1)))
    print("Correct answers: " + str(correct2) + " out of " + str(index - 1) + ".\n")

    # Output to a CSV file
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    csv_output_filename = ("csvs/" + rag1_name + "-vs-" + rag2_name + "-" + date_time_str + ".csv").replace(":", "_")
    llm_names = [rag1_name, rag2_name]
    llm_time_matrix = [time1_array, time2_array]
    llm_correct_matrix = [correct1_array, correct2_array]

    with open(csv_output_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["Model Name", "Q1 Time", "Q2 Time", "Q3 Time", "Q4 Time", "Q5 Time", "Q6 Time", "Q7 Time", "Q8 Time", "Q9 Time", "Q10 Time"])
        for name, matrix_row in zip(llm_names, llm_time_matrix):
            output_row = [name]
            output_row.extend(matrix_row)
            writer.writerow(output_row)

        writer.writerow(["Model Name", "Q1 Correct", "Q2 Correct", "Q3 Correct", "Q4 Correct", "Q5 Correct", "Q6 Correct", "Q7 Correct", "Q8 Correct", "Q9 Correct", "Q10 Correct"])
        for name, matrix_row in zip(llm_names, llm_correct_matrix):
            output_row = [name]
            output_row.extend(matrix_row)
            writer.writerow(output_row)

    # Show the graph
    # Line for LLM 1
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1 = time1_array

    # Line for LLM2
    x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y2 = time2_array

    # Plotting the lines
    plt.plot(x1, y1, label=rag1_name)
    plt.plot(x2, y2, label=rag2_name)

    # Add green or red circles for correct or incorrect answers
    index = 0
    for y in y1:
        if correct1_array[index] == "Y":
            plt.plot((index + 1), y, "go")
        else:
            plt.plot((index + 1), y, "ro")
        index = index + 1
    
    index = 0
    for y in y2:
        if correct2_array[index] == "Y":
            plt.plot((index + 1), y, "go")
        else:
            plt.plot((index + 1), y, "ro")
        index = index + 1

    # Adding legend, x and y labels, and titles for the lines
    plt.legend()
    plt.xlabel("Question Number")
    plt.ylabel("Time (s)")
    plt.title("LLMatchup Results")
    
    # Displaying the plot
    plt.show()

if __name__ == "__main__":
    main()