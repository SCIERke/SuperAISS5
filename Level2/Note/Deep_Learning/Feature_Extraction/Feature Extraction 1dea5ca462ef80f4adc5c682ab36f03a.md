# Feature Extraction

# Vector Representation

If we has data like this 

```bash
data = {
	{ "สุชาติ" : 170, 50 },
	{ "พี" : 150, 40 },
	{ "นานา" : 180, 60 },
	.
	.
	. 
}
```

If we want to compare who is the most similar to  พี between นานา ,สุชาติ or other , if we look at data in this format ,It may hard to do that. SO we can represent our data in vector-format instead and result would be like this:

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image.png)

It’s easier to compare right? we could simply find Euclid Distant between data point and we could easily compare it!!

Same method with this case too, we can represent color in vector format.

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image%201.png)

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image%202.png)

and easily find similarity between them.

But some cases, we need to get a **feature** (core data) of data first before use it to find similarity. 

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image%203.png)

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image%204.png)

For example, we need to embed text first before we use it to compare or other process such as we can use embedded text and represent it in vector-format and then we can find distant of it.

We will find that t**he distance between two words can be used to represent the difference between them when they share the same context but have different meanings.** For example:

- man - woman = king - queen

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image%205.png)

# Encoder

Transforms input data into a different (often lower-dimensional) representation. This representation is called an embedding or a latent representation.

For example:

if we draw 10*10 canvas it may like this

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image%206.png)

hard to interpret right? (may be it look like 1 or 7)

how about we draw it in 3*3

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image%207.png)

easier to interpret right? how about 1*1

![image.png](Feature%20Extraction%201dea5ca462ef80f4adc5c682ab36f03a/image%208.png)

i think now everyone draw same image (because it too small haha) so that why sometimes? encoder help us to easier represent our data!!