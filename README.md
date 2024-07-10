# Multi-Agent Approach Project

## Description
This project implements a multi-agent system using Python. It utilizes the `autogen` library for agent generation and interaction.

![alt text](https://github.com/mimmol99/MultiAgents/blob/main/MultiAgents_diagram.png?raw=true)

- The manager handles the user input and the interactions between agents.
- the Rag is based on Qdrant and retrieve top k similar chunks.
- the assistant answer the question using question and context.
- the evaluator check if the answer is coeherent with question and context.
  
 if the feedback is positive the answer is returned to the manager.
 
 if the feedback is negative the answer,the context and feedback is passed to the corrector.
 
- the corrector fix the asnwer using the feedback,the answer and the context.
- the refiner adjust the answer as a chatbot-like answer.
  
## Installation
```bash
pip install -r requirements.txt
```

## Usage

change the files_path variable value with the path of the documents to be retrieved

```bash
python3 main.py
```

Insert the input using the terminal and you will see the agents chat flow. Insert "exit" to exit the loop.

