
# WhatsApp RAG App 
This tutorial offers a comprehensive guide for building a QA Assistant or QA Chatbot that utilizes custom documents and is accessible to users via WhatsApp. This is accomplished using the LangChain framework for developing LLM applications. The tutorial integrates code and concepts from two sources: the [Twilio tutorial](https://www.twilio.com/en-us/blog/wikipedia-ai-assistant-whatsapp-python-langchain-openai) and an [excellent example](https://github.com/streamlit/example-app-langchain-rag) for creating a RAG-based app with LangChain.

## A Summary of Tools
Inorder to build this app, we will utilise the folowing tools and you need to make sure you get the required credentials and accounts.

### LangChain
[LangChain](https://www.langchain.com) is a powerful framework that allows developers to build applications powered by language models like the OpenAI commercial models (the GPT series) and open source models (e.g., LLama series). With LangChain with OpenAI, developers can connect to any LLM of their choice, bring in exteernal knowledge (data sources) such as websites, documents and create  create data-aware and agentic applications. This means that the AI assistant can connect with other data sources and interact with its environment effectively. In this example, the Chatbot will have access to custom documents which we will provide. 
### OpenAI and HuggingFace
In this tutorial, we will utilise OpenAI API to acces their suite of LLMs. In order to use OpenAI, we need access to the OpenAI developer API. However, the code is setup in such a way that you can also use [HuggingFace](https://huggingface.co) to connect to open source LLMs. 
### FastAPI
[FastAPI](https://fastapi.tiangolo.com) is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. It's designed to be easy to use and to provide automatic interactive API documentation. We will FastAPI to expose our app to Twilio WhatsApp framework.
### Twilio WhatsApp Messaging API
The [Twilio](https://www.twilio.com/en-us) WhatsApp Messaging API allows developers to send and receive messages using the WhatsApp platform through Twilio's API. This enables businesses to integrate WhatsApp messaging into their applications, providing a powerful channel for customer communication and engagement.This API will allow us to connect our LangChain Python app with WhatsApp and enable our users interact with our assistant in WhatsApp.
### PostgreSQL
[PostgreSQL](https://www.postgresql.org), often referred to as Postgres, is a powerful, open-source relational database management system (RDBMS). It has a strong reputation for reliability, feature robustness, and performance. We will use Postgres to log and store user conversations with our Chatbot.
### Ngrok
[Ngrok](https://dashboard.ngrok.com/get-started/setup/macos) is a tool that creates secure tunnels to your localhost, allowing you to expose a local development server to the internet. This is particularly useful for testing webhooks, APIs, and other web services that require public access during development. In this case, once the app is running locally (on your computer), Ngrok will allow other users to interact with the app.
### VS Code
In order to edit the Python code, it is recommended that you have a text editor. This can be any text editor but we recommend **[VS Code](https://code.visualstudio.com)**

In this example, we apply the knowledge we learned in the LangChain tutorial to develop a RAG based Chatbot which utilises documents from [WHO disease outbreak news](https://www.who.int/emergencies/disease-outbreak-news) to answer questions about latest disease outbreaks across the globe. 

## Setup Development Environment
Download this repository as a zipped folder or clone it if you are familiar with Git. Once you have the folder on your computer, navigate to that folder and execute the commands below.
### 1. Python virtual environment

Create a Python virtual environment to use for this project.
The Python version used when this was developed was 3.10.13. The code below creates a virtual environment and also installs all the Python packages we need for this tutorial

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you run into issues related to hnswlib or chroma-hnswlib while installing requirements you may need to install system package for the underlying package.

For example, on Ubuntu 22.04 this was needed before pip install of hnswlib would succeed.

```bash
sudo apt install python3-hnswlib
```

#### Setup .env file with API tokens needed.
Use VS code or any other text editor to create a ```.env``` text file and all the API tokens and API keys required.
```
# OpenAI
OPENAI_API_KEY="<Put your token here>"
# Hugging Face
HUGGINGFACEHUB_API_TOKEN="<Put your token here>"

# Twilio Credentials
TWILIO_ACCOUNT_SID="<Put your token here>"
TWILIO_AUTH_TOKEN="<Put your token here>"
TWILIO_NUMBER="<Put your token here>"

# PostgreSQL connection details
DB_USER = "<Put your token here>"
DB_PASSWORD = "<Put your token here>"

```
#### Verify environment
Perform a quick check for the core Python environment to make sure its running okay. 
1. Check that LangChain dependencies are working.

```bash
python basic_chain.py
```

2. Check that FastAPI setup is working
The output you will see is from an example document (a book by Bertrand Russel) just to check that the script is working.

```bash
uvicorn main:app --reload
```
> You should see no errors.

### 2. Configure Twilio Sandbox for WhatsApp
We will use the Twilio Sandbox for WhatsApp, which allows us to prototype immediately without waiting for the account to be approved by WhatsApp. Selected users can connect to this Sandbox WhatsApp number. The Twilio account can receive messages from WhatsApp users and forward them to our custom-made app, using the app’s application programming interface (API), which we will create using the FastAPI Python package.

To use Twilio's Messaging API to enable the chatbot to communicate with WhatsApp users, you need to configure the Twilio Sandbox for WhatsApp. Here's how to do it:
- Assuming you've already set up a new Twilio account, go to the Twilio Console and choose the Messaging tab on the left panel.
- Under Try it out, click on Send a WhatsApp message. You'll land on the Sandbox tab by default and you'll see a phone number "+14155238886" with a code to join next to it on the left and a QR code on the right.
- To enable the Twilio testing environment, send a WhatsApp message with this code's text to the displayed phone number. You can click on the hyperlink to direct you to the WhatsApp chat if you are using the web version. Otherwise, you can scan the QR code on your phone.
- To ensure that the Sandbox is working, complete all the steps and make sure they all run without errors 
- Now, the Twilio sandbox is set up, and it's configured so that you can try out your application after setting up the backend.
- Before leaving the Twilio Console, you should take note of your Twilio credentials: ```account-sid```, ```auth-token```, ```phonenumber``` and edit the ```.env```

### 3. Install PostgreSQL
Follow instructions on the [website](https://www.postgresql.org) to download and install Postgre on your system. Once installed, go on to perform the following actions:
- Use ```sudo``` rights to make sure you have added yourself as a user to Postgre. Please refer to the [documentation](https://www.postgresql.org/docs/current/tutorial-install.html)
- Take note of your ```username``` and ```password```
- Create a database ```createdb mydb``` where ```mydb``` can be a database name of your choice.
- Edit the ```.env``` file we created above and add your ```username``` and ```password```

### 3. Setup Ngrok
Once again, Ngrok is a development tool that you can use to expose a server running locally on a computer, possibly even behind a firewall, to the public Internet. Please use the [Ngrok](https://dashboard.ngrok.com/get-started/setup/macos) documentation to create an account, download and install Ngrok on your computer and test that its running. You can lso refer to this short [blog post](https://www.twilio.com/en-us/blog/using-ngrok-2022) if you need further details about setting up Ngrok.


## Project Setup 
The repository contains the following key directories.
> **data.** This folder contains the documents which the Chatbot will use.
> 
> **store.** The Chroma vector database files are stored in this directory
> 
> **examples.**. This folder contains example documents which can be used in case the data folder doesnt have any

Note that this an advanced version of what we covered in the LangChain tutorial. All the different components of creating Chatbot such as creating chains, memory management, loading and splitting documents is handled in seperate Pythin modules (files). Although this looks complicated, its the best practice for deploying Python apps in the wild. 

The following Python scripts are the important ones to understand. Feel free to read and explore what the other scripts are doing.
>**main.py** This contains the FastAPI function which deploys the API on your computer.
>
>**local_loader.py** This script has LangChain functions which are loading the PDF files and get them ready for storage in vector database.
>
>**full\_chain.py/basic\_chain.py** These too files contains the LangChain prompts logic which creates the Chatbot and accesses the documents in the ```data``` folder.

## Configure your database
In this tutorial, you will the Python package SQLAlchemy to access the PostgreSQL database. So the goal is to store conversations for your app in the PostgreSQL database. The code perfoming this task is in the ```model.py``` file. Note that here, you've used decouple.config module to access the environment variables for your database: DB_USER and DB_PASSWORD

## Building the LangChain Agent
In this project, there are several Python scripts which are working together to create a LangChain Chatbot as follows.
1. **Loading and splitting documents.**
2. **Creating vector database.**
3. **Setting up Chatbot Memory.**
4. **Creating LLM Chains.**

## Putting it All Together
### 1. Setup FastAPI backend.
- We create a simple FastAPI application which handles a single incoming request. The code for this is in the ```main.py``` file.
- Make sure that you successfully configured Twilio Sandbox in the step above. In this step, we will use the Twilio Python package to create a function that can send a WhatsApp message. This code is embedded in the **reply** function in  ```main.py``` file. 
### 2. Running the FastAPI app
In order to run the main function, use the code below
```
uvicorn main:app --reload
```

## Example Queries for the Chatbot
The Chatbot will answer questions on health outbreaks as announced by WHO

### Question 1: Check whether there is an outbreak in a country
**Question:** Is there a disease outbreak in Zambia?

**Answer:**
Yes, there is currently an outbreak of anthrax in Zambia, particularly in the Sinazongwe district, which is the epicenter of the outbreak. The outbreak has spread to several provinces in the country, affecting both animals and humans.

**Question:** What about Tunisia?

**Answer:**
I don't have information about a current disease outbreak in Tunisia. If you have any specific concerns or need more details, I recommend checking the latest updates from official health sources or the World Health Organization for the most accurate information.

### Example 2: Details about a disease outbreak

**Question:**
When did anthrax start in Zambia?

**Answer:**
The anthrax outbreak in Zambia began on 1 November 2023 when the IHR National Focal Point of Zambia notified WHO of the outbreak in humans.

**Question:**
Has anyone died due to anthrax in Zambia?

**Answer:**
Yes, there have been sporadic suspected cases and deaths reported in different districts and provinces of Zambia since June 2023 due to the anthrax outbreak. The epidemic is spreading along the provinces located along specific rivers, and the Bacillus anthracis bacteria can form highly resistant spores that survive in the environment for decades. The risk for human fatalities is present, especially in cases of severe illness that can lead to respiratory failure and other complications.


## References
This app was adapted from the [Streamlit example app](https://github.com/streamlit/example-app-langchain-rag).


## Misc Notes

- There is an issue with newer langchain package versions and streamlit chat history, see https://github.com/langchain-ai/langchain/pull/18834
- This one reason why a number of dependencies are pinned to specific values.
