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

$$Pr(Y = j | X = x_0) = {1 \over K} \sum_{i \in N_0}I(y_j = j)$$

The classifier result will be the max probability: $argmax_j(Pr)$

$$C^{KNN}(x) = argmax_j({1 \over K} \sum_{i \in N_0}I(y_j = j))$$

Small K values lead to higher variance, $K=1$ will perfectly fit the training data.

## 4.2 Naive Bayes classifier

$${\displaystyle C^{\text{Bayes}}(x)={\underset {y_i}{\operatorname {argmax} }}\operatorname {P} (Y=y_i)\prod _{j} P(X_j|Y=y_j)}$$

## 4.3 Logistic regression

In logistic regression we model the probability of an observation belonging to one of two classes with logistic function. \output ranges between 0 and 1 (<b>Figure 4.1</b> left side)

$$p(X) = {e^{\beta_0 + \beta_1X_1 + ... +  \beta_pX_p} \over 1 + e^{\beta_0 + \beta_1X_1 + ... + \beta_pX_p}}$$

We can transform the above to odds form $p \over 1-p$

$p = {e^z \over 1 + e^z}$<br>
$p(1 + e^z) = e^z$<br>
$p + p e^z = e^z$<br>
$p = e^z(1-p)$<br>
${p \over 1 - p} = e^z$

Giving

$${p(X) \over 1 - p(X)} = e^{\beta_0 + \beta_1X_1 + ... + \beta_pX_p}$$

Taking $log$ of both sides gives the log odds or **logit**

$$log\bigg({p(X) \over 1 - p(X)}\bigg) = \beta_0 + \beta_1X_1 + ... + \beta_pX_p$$

Which is a linear function, see right side of **Figue 4.1**

<p align="center">
<img src="./img/04-log-function.png" width="400">
<b>Figure 4.1: </b><i>Left side probability p, rights side logit transformation. Observations move from 0 to negative infinity and from 1 to infinity</i> (source StatQuest)
</p>

We can use categorical variables trough dummies, same as linear regression.

### 4.3.1 Fitting the model

The logistic function can be fit using maximum likelyhood. The lokelyhood function is

$$\ell(\beta_0, \beta_1) = \prod_{i:y_i=1}p(x_i)\prod_{j:y_j=1}\big (1 - p(x_j)\big )$$

### 4.3.2 Assessing the model

Each estimated coefficient has associated *z*-statistic
$$\hat \beta_1 \over \operatorname{SE}(\hat \beta_1)$$

If *z*-statistic is large, and the associated $p$-value is below a selected $\alpha$ we can reject the null hypothesis: $H_0: \beta_1 = 0$