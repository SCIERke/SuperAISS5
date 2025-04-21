# RAG (Retrieval-Augment Generation)

# The problems?

Nowadays, Generative AI stepping up to important role. We use it in many ways, such as a chatbot for customer Q&A or as our best copilot for handling difficult tasks . However, one of the main problems we are most concerned about is “Hallucination”.

Hallucination is behavior that AI give us information that might appear correct, but it is actually completely wrong or purely conjecture. (The reason is transformer **predict the next word/token** based on patterns in their training data, not actual facts and  some transformers create random tokens so each question our answer would be not same)

and another main problems is **Bias** of model. Some models train with bias information that lead to bias result.

In fact, there is other problems that we will be faced when we’re training.

 

# How can we solve these problems?

There are many ways to solve these problems but we will focus to main solution that we mostly using it which is **RAG!!!** ,Our hero of this note.

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image.png)

# RAG

Normally, when we query LLM, we would only give it question and AI will answer our question base on only question **which don’t understand any context of our problem at all!!.** 

This is the problem: imagine a situation where I ask about Thai driving laws. On the surface, the AI might handle it well, but when it comes to deeper, more specific questions, it might start hallucinating — because it hasn’t been trained on that specific data and doesn't truly understand the context.

So This is where RAG born. RAG help us to find context of the question by query the **exist** information that may be in database or documents and give it to AI with question.

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%201.png)

# Type of RAG

There are many type of RAG. In this note, I will separate type of it by format of data that we’re faced which will have:

- Structured RAG
- Unstructured RAG

 

# Structured RAG

**Structured data:** using prompt design to create SQL code for querying result from a
database

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%202.png)

The process is we will use SQL Agent to create SQL query to search context information and attach it with our query to send it to LLM.

which we can easily create by LangChain framework. see tutorial here:

 https://python.langchain.com/docs/tutorials/sql_qa/ 

or [super ai ss5 tutorial](https://colab.research.google.com/drive/1BEST7HrkA5MDJjkn)

# Unstructured RAG

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%203.png)

**Unstructured data:** using embedding model to transform text and the resulting
vector through similarity search with a vector database

For unstructured data, we will translate our data into the format that model understand and easy to search which has 2 ways:

- Dense Vector Search (embedding)
- Sparse Vector Search (TF/IDF)

## Dense Vector Search

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%204.png)

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%205.png)

The process of DVS start by loading our source and then we will split the text into small, semantically meaningful chunks. After we process the data, we will embed it by embedded model. 

(HERE IS  [benchmark of Thai Sentence Vector Benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark))

And we will store embedded data in vector format and we can use it later in query process.

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%206.png)

We can store our vector in place which we called vector store. Here is a list of vector store:

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%207.png)

## Sparse Vector search

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%208.png)

In SVS, process will similarly to DVS but we will change data into TF-IDF forma and store it. When we need to query it , we will match queries to encoded data by BM25 model.

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%209.png)

And there is more effective ways to RAG data by combine DVS and SVS together called “**Hybrid Search”**

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%2010.png)

And if we got too much relevant documents from this ways we can rank our data base on their relevance ,This process called **Rerank.**

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%2011.png)

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%2012.png)

## Rank of methods

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%2013.png)

source: [https://infiniflow.org/blog/best-hybrid-search-solution](https://infiniflow.org/blog/best-hybrid-search-solution)

## Code

[Here is a example code of all method that i said.](https://colab.research.google.com/drive/1-)

## Standard RAG

here is standard RAG method that Anthropic guide.

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%2014.png)

## Others way

There is more advance way that i will not mention.

### HyDE

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%2015.png)

source: [here is source](https://www.notion.so/RAG-Retrieval-Augment-Generation-1dca5ca462ef80148997c42adcf2c69c?pvs=21)

### Fusion RAG

![image.png](RAG%20(Retrieval-Augment%20Generation)%201dca5ca462ef80148997c42adcf2c69c/image%2016.png)

source: [medium](https://www.notion.so/RAG-Retrieval-Augment-Generation-1dca5ca462ef80148997c42adcf2c69c?pvs=21)