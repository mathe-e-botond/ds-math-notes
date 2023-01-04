# 4. Classifiers

Classification is the problem of mapping variables to a categorical dependent variable. While in some cases there can be more than two categories, we can reduce the problem of classifing to a category or another and repeating for the latter. 

We can measure performance of classifier trough the error rate

$${1 \over n} \sum_{i=1}^nI(y_i \ne \hat y_i)$$

where $\hat y$ is predicted class for $i$ th observation and $I(y_i \ne \hat y_i)$ is an indicator having value of $0$ in case of missclassification and $1$ for correct classification.

The test error rate is minimized by maximizing the **Bayes classifier**, which assigns each observation to the most likely class, given it's predictor values

$$argmax_j(Pr(Y = j | X = x_0))$$

We use $argmax$, because we need the max $j$ parameter and not maximum probability.

Prediction of the Bayes classifier is determined by the so called **Bayesian decision boundary** where probability is $0.5$. The Bayes classifier produces the lowest error rate called **Bayes error rate**

$$1 - E(max_j(Pr(Y = j | X)))$$

The Bayes error rate is analogous to the irreducible error of linear models. The Bayes classifier in most cases is unknown and we would like to estimate it.

Proofs:
https://en.wikipedia.org/wiki/Bayes_classifier

## 4.1 K nearest neighbour classifier (KNN)

KNN classifier tries to estimate the Bayes classifier, by finding the K neaerest observation in training data closest to $x_0$ test observation

$$Pr(Y = j | X = x_0) = {1 \over K} \sum_{i \in \Nu_0}I(y_j = j)$$

The classifier result will be the max probability: $argmax_j(Pr)$

$$C^{KNN}(x) = argmax_j({1 \over K} \sum_{i \in \Nu_0}I(y_j = j))$$

Small K values lead to higher variance, $K=1$ will perfectly fit the training data.

## 4.2 Naive Bayes classifier



$${\displaystyle C^{\text{Bayes}}(x)={\underset {y_i}{\operatorname {argmax} }}\operatorname {P} (Y=y_i)\prod _{j} P(X_j|Y=y_j)}$$

## 4.3 Logistic regression

