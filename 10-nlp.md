# **10. Natural language processing**

**Natural language processing** or **NLP** for short the field of AI concerned with working with natural languages like English (as opposed to programming language like Python) is one of the fastest growing fields of AI in recent years.

## **10.1 Common NLP terms and statistics**

**Tokenization** breaks a text into tokens, in most cases words. In English punctuation and white spaces can be used. Sentence ending punctuation should be unified to a token signifying sentence boundary. In some languages like Chinese, where spaces are not used as word separator, dictionaries are required for tokenization. Some special types of tokenizers:
* **Contraction** is a form of tokenization mainly for grammatical structures like `wasn't`, splitting into `was` and `n't` or when possible to it's full form of `was` and `not`, so each new token has a meaning on their own. THis helps to reduce the number of tokens and also to covey more meaning to the model.
* **Casual tokenization** when we want to work with social media content, it's often required to remove handlers (e.g. `@username`), emoticons, character repetitions (e.g `loool` can be normalized to `lol`)

**Normalization** can be used to unify tokens with same meaning. Search engines might apply stronger normalization, to increase **recall** metric as the number of matches against (possibly much bigger reduction in) **precision**. For NLP pipelines normalization might reduce too much context so might be reduced or skipped entirely. Common normalization techniques are:
* **Case folding**: unifies upper and lower case, which might be applied to sentence starting words only but skipped for proper nouns, like names. 
* **Ascii folding**: unifies special characters to more common forms, e.g in languages which use accents like `á` can be normalized to English letters `a`.
* **Lemmatization** using part of speech, can convert words entirely to other, more common words with same meaning e.g `better` as an adjective might be normalized to `good`
* **Stemming** normalizes words to their stem, removing grammatical structures like plural form. Usually applied after lemmatization, more simple to implement compared to lemmatization but also reduces more meaning. This might be achieved with a complex language specific rule set, e.g `spelling` simply remove `ing`, `handling` remove `ing` but restore the missing `e`, the `ing` in `ping` is not a grammar form so cannot be removed. Another option is to use statistical tools. 

**Part of speech (PoS) tagging** 

**Named entity recognition**

**Syntactic parsing**



### **10.3.2 TF-IDF**

A common statistic or scoring used with natural language documents is the **term frequency, inverse document frequency** or TF-IDF. The primary use case of TF/IDF is document retrieval based on a search term. Given a set of documents, we can calculate TF/IDF score for each document $d$ in our set for the search term $t$. TF-IDF can be broken down as a relationship between two other statistics, TF and IDF:

$$\operatorname{TF-IDF} = \operatorname{TF} \cdot \operatorname{IDF}$$

The TF term for a document ($d$) given the term ($t$) is defines how many times a word appears in a document as an importance between the document and the term.

$$\operatorname{TF}(t,d) = {count(t\ in\ d) \over {count(*\ in\ d)}}$$

According to **Zip'f law** most languages follow a pattern. If we count the term frequencies and rank them in decreasing order, the frequency of any word is inversely proportional to it's rank. Given a large enough corpus in English, the first term would be `the` occurring 2 times as `of` at rank 2, 3 times as `to` at rank 3, 4 times as `a` at rank 4, etc. Other languages will follow a similar pattern.

The inverse document frequency defines how common the term is in terms of the corpus. Words which appear frequently in all documents will get a higher IDF score compared to words that are unique to some documents. For large scale corpus the IDF term can explode, so commonly the natural log is used to scale:

$$\operatorname{IDF}(t) = ln\bigg({N \over count(t\ in\ D)}\bigg)$$

where $D$ is the set of all documents and N is the total number of documents. The choice for natural log instead of other base is mainly for consistency. If the term $t$ does not exist in $D$, a division by $0$ can occur. A common solution is to add $+1$ to the denominator:

$$\operatorname{IDF}(t) = ln\bigg({N \over count(t\ in\ D)  + 1}\bigg)$$

A similar result would be provided if we apply a technique called **Laplace smoothing**.

TF-IDF is commonly used as a document retrieval in document stores. A commonly used document store is ElasticSearch which uses BM25, an extended version of TF/IDF to rank documents given a list of search terms or tokens. For each token and each document a TF/IDF score can be computed. The TF-IDF scores resulting in the case of multiple tokens can  be combined with a simple summation or weighted summation. The resulting combined TF-IDF scores for each document determines the most relevant documents for the search terms.

## **10.3 Primitive language models**

Before the age of transformers a set of simple models were proposed. While these models are not too accurate and less used today, they serve the basis of the deep learning models.

### **10.3.2 Bag of words**

The **bag of words** (BoW) model is a common representation used in NLP. We can use it to represent a sentence, usually part of a set of sentences or corpus, as a vector. The steps are as follows:

1. Assign all terms from our corpus to an index from $1$ to $|V|$, where $|V|$ is the total number of unique terms in our corpus ($V$ is the set of terms).
2. Construct a vector of size $|V|$
3. For each element at index $i$ assign<br>
    a.) $1$ if the word $i$ is in the sentence, $0$ otherwise - called binary BoW<br>
    b.) Number of occurrences of the word $i$ in the sentence, also called term frequency or TF<br>
    c.) TF/IDF score of the word $i$, TF would capture how important the term is for the sentence, IDF would measure how important the word is considering the entire corpus<br>
    d.) normalized frequencies - number of times the word $i$ occurs in the sentence divided by the total number of words in the sentence. This normalization ensures that longer documents do not have an inherent advantage over shorter ones. 

BoW vectors are very sparse so not too practical, but they can already be used with vector arithmetics. The dot product of two binary BoWs would give a similarity measure between sentences in terms of how many words repeat across the two sentence.

In most cases we want to measure if documents have similar words in similar counts as a measure of distance. In the $|V|$ dimensional space where every word is a dimension, the best measure for this is the angle between two BoW vectors. We can quantify the distance as an angle with **cosine similarity**:

$$cos\ \Theta = {A \cdot B \over |A| |B|} = {\sum_{i=1}^{|V|} a_ib_i \over \sqrt {\sum_{i=1}^{|V|} a_i^2} \sqrt {\sum_{i=1}^{|V|} b_i^2}}$$

where $A$ and $B$ are two BoW vectors, $A \cdot B$ is the dot product, $|A|$ and $|B|$ are the L2 norms of the vectors. The above expression might be more familiar in the form of $A \cdot B = |A| |B|\ cos\ \Theta$. Cosine similarity ranges between $-1$ and $1$, where 1 means that the two vectors point in the same direction, but their magnitude might be different. Since term frequencies cannot be negative, for BoWs the minimum cosine similarity would be 0, which happens for perpendicular vectors, meaning that no common word is being used in two sentences.

### **10.3.3 Naive Bayes**

**Naive Bayes** can be used to classify a text to some category, given a training data set. in the context of NLP it's used a a non-parametric estimator. From Chapter 5, the Naive Bayes classifier is as follows:

$${\displaystyle C^{\text{Bayes}}(x)={\underset {k}{\operatorname {argmax} }} P (Y=k)\prod _{j} P(X_j = x_j|Y=k)}\tag{10.1}$$

In the context of NLP the class is usually a category. This model is also used for sentiment analysis, where each sentiment is treated is a category. For example given a list of product reviews (e.g app store reviews) with both a text and a five star review, a Naive Bayes model can be constructed to also evaluate tweets about the product without explicit ratings. 

Given a sentence $x$ we want to classify it to a category $k$ using Naive Bayes: $\displaystyle C^{\text{Bayes}}(x)$. Every word in the sentence acts as a predictor $x_i$.  Naive Bayes assumes that the probability of a combinations of words (or the lack of words) defining the class of sentence is same as the probability for each word separately doing the same. In most cases, this is not true, e.g *goal* will have a different meaning if it's combined with the word *career* and different meaning if occurs together with *soccer*. 

To apply Naive Bayes a training dataset is needed, also called **corpus** in the context of NLP. The term $P(Y=k)$ called the prior, can be calculated as the ratio of training sentences of class $k$, noted with $n_{\operatorname{x} \rightarrow k}$, divided by number of total training samples $N$

$$P(Y = k) = {n_{\operatorname{x} \rightarrow k} \over N}$$

For each word in the training dataset, we can create a distribution of how likely each word predicts a category as the ratio between the number of occurrences of the word in sentences of class $k$, noted as $n_{\operatorname{x_i} \rightarrow k}$, divided by the total number of words in sentences of class $k$, noted with $n_{\operatorname{x_* \rightarrow k}}$:

$$P(X = x_i | Y = k) = {n_{\operatorname{x_i} \rightarrow k} \over n_{\operatorname{x_* \rightarrow k}}}$$

This probability is also referred to as likelihood.

When we get a new sentence we can predict the category by plugging in the prior and likelihood for each word calculated using the training data, into 10.1. 

If the test sentence has a word that does not appear in a category in the training set, the likelihood term will be $0$ and the probability for that category will also become $0$ irrespective of the likelihood of other words in the sentence. To counter this we can change the likelihood as 

$$P(X = x_i | Y = k) = {n_{\operatorname{x_i} \rightarrow k} + q \over n_{\operatorname{x_* \rightarrow k}} + Nq}$$

where $q$ is a constant, usually with a value of 1. This will result as minimum likelihood of $1 \over N$.

The BoW can be used as an input to "train" the Naive Bayes model. We can create a single BoW for each category, where each index $i$ would capture how many sentences contains that word. 

### **10.3.4 N-gram model**

In the context of index creation for document retrieval we usually talk about n-grams in the context of characters. In the string `abc` the possible 2-grams are `ab` and `bc`. Edge n-grams from left side are `a`, `ab` and `abc`.

In the context of NLP, the n-gram model is defined in context of words. We are trying to capture groups of words that occur together in a specific order. This information was last in BoW and Naive Bayes models. 

Too infrequent (lyrical word combination) and too frequent (e.g. `at the`) n-grams are not useful for modelling and might contribute to overfitting. The simplest way to get the useful n-grams is to apply **rare token** (exclude too rare n-gramns) and **stop word** filters (exclude too common n-gramns). Stop word filters should be applied to n-grams and not the stop words themselves, since n-grams can capture relationship between stop words and other words (4 grams might be needed for this).

## **10.3 Deep learning in NLP**