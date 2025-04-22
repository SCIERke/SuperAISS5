# AI / ML Overview

For this note I’m gonna skip some topics because I’m too lazy to lecture it all!

# Machine Learning

Machine Learning is a method that reverse from traditional method.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image.png)

If you look at this diagram, you will find that Normally we will put data + program and computer will make output for you but ,in machine learning, we will put data + output and hope computer to make program for us. 

# EDA

# Deep Learning

Deep Learning is subset of ML which bypass some process such as feature extraction, classification, etc.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%201.png)

We usually use DL when we don’t have domain knowledge about datasets because DL automatically extract feature for us but cons of using DL is it black-box methods.

We can’t control entire process and another cons is, It requires a high computational cost.

There is a bunch of problem that we will face when we use it. So you need to be careful!!

# ML/AI capabilities

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%202.png)

# Type of ML

There is many ways to classify model but reference from mentor in super ai, we can separate model like this.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%203.png)

# ML Workflow

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%204.png)

# ML Life Cycle

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%205.png)

# Feature Selection

We should clarify how should the input data be represented.
• How do we extract features from raw sources?
• Consider to include domain experts to specify what data aspects
are most important for the particular ML task.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%206.png)

# Evaluation Metrics

Evaluation is most crucial process for create model. Good evaluation lead to good result because good evaluation can show that how exactly our model is it.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%207.png)

# Model Serving Patterns

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%208.png)

# GPU Vs CPU

Nowadays, GPU become crucial role of training model. Knowing when and how to use GPU is important and understanding how is work also important. Here is comparing chart of CPU and GPU 

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%209.png)

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%2010.png)

# Cloud

- Massive Compute Power
- Scalability and Elasticity: Cloud infrastructure can elastically scale
resources up or down based on AI workload demands.
- Cost Efficiency: Cloud’s pay-as-you-go model reduces capital expenses
and makes advanced AI capabilities accessible to both large enterprises and
smaller organizations. (???)
- Collaboration and Accessibility
- Automation and Integration
- Hybrid and Multi-Cloud Flexibility

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%2011.png)

# Tool for managing life cycle of ML

## ML Flow

MLflow is **a versatile, expandable, open-source platform for managing workflows and artifacts across the machine learning lifecycle**. It has built-in integrations with many popular ML libraries, but can be used with any library, algorithm, or deployment tool.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%2012.png)

## Kubeflow

Kubeflow is a community and ecosystem of open-source projects to address each stage in the machine learning (ML) lifecycle with support for best-in-class open source tools and frameworks. Kubeflow makes AI/ML on Kubernetes simple, portable, and scalable.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%2013.png)

## Data Version Control (DVC)

DVC help us to version our data and model which will help us to experiment method that will impact to our data. We can summarize DVC to this:

- Version your data and models. Store them in your cloud storage but keep their version info in
your Git repo.
- iterate fast with lightweight pipelines. When you make changes, only run the steps impacted
by those changes.
- Track experiments in your local Git repo (no servers needed).
- Compare any data, code, parameters, model, or performance plots.
- Share experiments and automatically reproduce anyone's experiment.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%2014.png)

# Trend about AI in 2025

## Generative AI

Nowadays,We mostly use AI in our daily life. Whether it's using AI to generate images, videos, or answer general questions.

# Large Language Model (LLM)

A large language model is an AI system trained on vast amounts of text data to understand
and generate human-like text for various language tasks, though it can sometimes produce
inaccurate or biased outputs. here is timeline of LLM from past to present.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%2015.png)

# LLM - OMNI (Multimodal)

Multimodal AI is model that can use from ALL to ALL → Text to Text, Text to Image, Text to Video, etc.

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%2016.png)

# Wrap Up

![image.png](AI%20ML%20Overview%201dda5ca462ef80f7b4dac2c6122ea33b/image%2017.png)