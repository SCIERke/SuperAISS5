# Transformer

# Attention

## Basic Retrieval Mechanism

In Basic Retrieval Mechanism, you may keep your data key-value format right? so component of it would be like:

- query
- key
- value

and mechanism would be like  

- Find a key that matches with query
- Return only one value of that key

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image.png)

We can use it with word embedding to keep representation vector-format in table like this?

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%201.png)

and that may lead to problem because as you know school won’t be same meaning in every context. so that’s origin of **Attention.** Yet the embedding is **always the same**. So fixed word embeddings fail in understanding *contextual nuance*.

## Our hero Attention!

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%202.png)

This is exactly where **Attention** came in — and led to models like **Transformer** (which powers BERT, GPT, etc).

With **attention**, the model can:

- Look at *all* the words in a sentence.
- Dynamically weigh how much attention to give each one.
- Adjust the meaning of a word based on the surrounding context.

So now, **“school” in one sentence has a different vector than in another**, thanks to dynamic contextualization.

and there is many similarity comparison, below chart is samples of similarity functions:

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%203.png)

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%204.png)

and we can apply softmax to it to make it easier to compare. 

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%205.png)

## Position Encoding

Sentence like “I eat cat” and “Cat eat I” might be same right,this problem make LLM hard to understand our sentence context. **Luckily,** **Position Encoding can help us,** It uses this regulation to determine the context of sentence/word

- No sequential order in attention layer
- Adding the information about the position into
embedding vector

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%206.png)

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%207.png)

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%208.png)

# Layer Normalization

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%209.png)

# Multihead Self-Attention: MSA

![image.png](Transformer%201dea5ca462ef8074a96bd8a5697259f9/image%2010.png)