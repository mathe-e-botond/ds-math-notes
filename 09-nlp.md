# **9. Natural language processing**

**Natural language processing** or **NLP** for short the field of AI concerned with working with natural languages like English (as opposed to programming language like Python) is one of the fastest growing fields of AI in recent years.

## **9.1 Common NLP terms and statistics**


### **9.3.2 TF-IDF**

A common statistic or scoring used with natural language documents is the **term frequency, inverse document frequency** or TF-IDF. The primary use case of TF/IDF is document retrieval based on a search term. Given a set of documents, we can calculate TF/IDF score for each document $d$ in our set for the search term $t$. TF-IDF can be broken down as a relationship between two other statistics, TF and IDF:

$$\operatorname{TF-IDF} = \operatorname{TF} \cdot \operatorname{IDF}$$

The TF term for a document ($d$) given the term ($t$) is defines how many times a word appears in a document as an importance between the document and the term.

$$\operatorname{TF}(t,d) = {count(t\ in\ d) \over {count(*\ in\ d)}}$$

The inverse document frequency defines how common the term is in terms of the corpus. Words which appear frequently in all documents (e.g "the", "a", "of", etc) compared to words that are unique to some documents will get a higher IDF score. For large scale corpus the IDF term can explode, so commonly the natural log is used to scale:

$$\operatorname{IDF}(t) = log\bigg({N \over count(t\ in\ D)}\bigg)$$

where $D$ is the set of all documents and N is the total number of documents.

TF-IDF is commonly used as a document retrieval in document stores. A commonly used document store is ElasticSearch which uses BM25, an extended version of TF/IDF to rank documents given a list of search terms or tokens. For each token and each document a TF/IDF score can be computed. The TF-IDF scores resulting in the case of multiple tokens can  be combined with a simple summation or weighted summation. The resulting combined TF-IDF scores for each document determines the most relevant documents for the search terms.

## **9.3 Primitive language models**

Before the age of transformers a set of simple models were proposed. While these models are not too accurate and less used today, they serve the basis of the deep learning models.

### **9.3.2 Bag of words**

The bag of words model is a common representation used

### **9.3.3 Naive Bayes**

**Naive Bayes** can be used to classify a text to some category, given a training data set. in the context of NLP it's used a a non-parametric estimator. From Chapter 5, the Naive Bayes classifier is as follows:

$${\displaystyle C^{\text{Bayes}}(x)={\underset {k}{\operatorname {argmax} }} P (Y=k)\prod _{j} P(X_j = x_j|Y=k)}\tag{9.1}$$

In the context of NLP the class is usually a category. This model is also used for sentiment analysis, where each sentiment is treated is a category. For example given a list of product reviews (e.g app store reviews) with both a text and a five star review, a Naive Bayes model can be constructed to also evaluate tweets about the product without explicit ratings. 

Given a sentence $x$ we want to classify it to a category $k$ using Naive Bayes: $\displaystyle C^{\text{Bayes}}(x)$. Every word in the sentence acts as a predictor $x_i$.  Naive Bayes assumes that the likelyhood of a combinations of words (or the lack of words) defining the class of sentence is same as the likelyhood for each word separately doing the same. In most cases, this is not true, e.g *goal* will have a different meaning if it's combined with the word *career* and different meaning if occurs together with *soccer*. 

To apply Naive Bayes a training dataset is needed, also called **corpus** in the context of NLP. The term $P(Y=k)$ can be calculated as the ratio of training sentences of class $k$, noted with $n_{\operatorname{x} \rightarrow k}$, divided by number of total training samples $N$

$$P(Y = k) = {n_{\operatorname{x} \rightarrow k} \over N}$$

For each word in the training dataset, we can create a distribution of how likely each word predicts a category as the ratio between the number of occurrences of the word in sentences of class $k$, noted as $n_{\operatorname{x_i} \rightarrow k}$, divided by the total number of words in sentences of class $k$, noted with $n_{\operatorname{x_* \rightarrow k}}$:

$$P(X = x_i | Y = k) = {n_{\operatorname{x_i} \rightarrow k} \over n_{\operatorname{x_* \rightarrow k}}}$$

If we plug into (9.1) these results, we can calculate the probability of each class for an input sentence.

## **9.3 Deep learning in NLP**