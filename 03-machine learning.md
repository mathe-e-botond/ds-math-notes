# 3. Machine learning

Machine learning is a set of techniques where we try to fit a model to an observed data set. There are two main objectives why we would like to do this:
* Create simple models that don't fit the data too well but can provide explanations about relationship of variables, which we call **inference**.
* **Forecast** the an unseen value of a variable using past values of itself or other variables.

The variable we would like to model or forecast is called the **independent** variable. The variables we use to model are called **dependent** variables. We can model an **independent** $Y$ variable with the **dependent** variables $X$ of a sample of observations. We assume the following relationship

$$Y = f(X) + \epsilon \tag{4.1}$$

We define estimate in the form

$$\hat{Y} = \hat{f}(X) \tag{4.2}$$

$\epsilon$ is the error term and can be decomposed using the expected value of (4.1) and (4.2):

$$E[(Y-\hat{Y})^2] = \underbrace{[f(X)-\hat{f}(X)]^2}_\text{Reducible error} + \underbrace{\text{Var}(\epsilon)}_\text{Irreducible error}$$

Proof:

Using (4.1) and (4.2)<br>
$E[(Y-\hat{Y})^2]=E[(f(X)+\epsilon-\hat{f}(X))^2]$<br>
$=E[(f(X)-\hat{f}(X))^2+2 \epsilon (f(X)-\hat{f}(X)) +\epsilon^2]$

Because the expectation is linear<br>
$=E[(f(X)-\hat{f}(X))^2] +2E[\epsilon (f(X)-\hat{f}(X))] +E[\epsilon^2]$

Because the expectation of $f$ and $\hat{f}$ are constant<br>
$=[f(X)-\hat{f}(X)]^2 +E[\epsilon^2] +2E[\epsilon (f(X)-\hat{f}(X))$

Because the mean of $\epsilon$ is zero<br>
$=[f(X)-\hat{f}(X)]^2 +E[\epsilon^2]$

Because the variance of $\epsilon$ is $E(\epsilon^2)$<br>
$=[f(X)-\hat{f}(X)]^2 + \text{Var}(\epsilon)$

We can opimize our estimate to minimize reducible error but irreducible error is also unknown and our model might overfit by including some fit on the noise to our estimate.

When we would like to fit a model to the data we first assume a model structure, this process is called **model choise** e.g. a linear model, tree model, nerual network, etc. We then try to estimatee the model parameters. There are two main parameters:
* **Hyperparameters**: describe the structure of model or model fitting, e.g. learning rate, layers in a neural network, the K in KNN, etc
* **Weights**: are free parameters we can adjusted during the learning process to best fit the model to the observed data

Estimating hyperparameters and weights usually have different techniques.

## **3.1 Model fitting**

There are two main probabilistic optimization frameworks to estimate model parameters, also commonly referred to as **weights**, given a set of observation: **Maximum Likelihood Estimation** (MLE) and **Maximum a Posteriori** (MAP). The difference is that MAP assumes a prior probability distribution and tries to estimate parameters using the posterior probability, MLE estimates parameters using the prior based on observations only.

$$\theta_{MLE} = argmax_{\theta}\ f_n(x_1...x_n|\theta)$$

If values of $x_1...x_n$ are i.i.d or we assume it, becomes

$$\theta_{MLE} = argmax_{\theta}\ \prod_{i=1}^nf(x_i|\theta)$$

We than try to optimize $L(\theta)$. Since it's an optimization problem, we can optimize log likelyhood of $log\ L(\theta)$ instead to facilitate derivative calculations and avoid underflows due to several products of small decimal values.

If we assume a prior distribution in addition to our observations, we can apply MAP, which maximizes the posterior function :

$$
\begin{aligned}
\theta_{MAP} &= argmax_{\theta}\ f(\theta|x_1...x_n) \\
&= argmax_{\theta}\ g(\theta) f(x_1...x_n|\theta)
\end{aligned}
$$

We skipped the denuminator (so-called marginal likelihood) after applying the Bayes rule above because it does not change the optimization problem.

For example for linear regression, MLE estimates the mean squared loss, applying MAP will estimate L2 regularization as well.

There are two main methods of model fitting
* If there is closed solution for the optimization we can apply analytical calcuation. This is only possible in few cases, for simple models with few parameters
* Iterative approach: a more commonly used approach, which can fit very complex models

## **3.2 Cost function and bias-variance trade-off**

To measure how well the model fits our observed data we can use a **cost function**. For a function to be considered as a cost function, it needs to fulfill the following attributes

* Should always be positive
* If our estimate improves, the cost function should decrease

Using the likelyhood function, which is the probability we can observe our data given our model, we can transform it to be a positive function, which decreases the better the fit. This is called the **negative log likelyhood cost function**. Given a set of observations $X$ and a statisticam model with parameters $\theta$ and the likelyhood function $L(\theta | X)$, the cost function is:

$$\operatorname{NLL} = - \ln(L(\theta | X))$$

The likelyhood function is mainly used to estimate parameters of proability distribution given the observed data, but might not have closed form or might have more than one local minima which is why other cost functions which are easier to optimize might be used to fit machine learning models.

A popular example of a cost function is the *mean squared error* or MSE, which is the averagee of the squared difference of predicted and actual output for each observation $i$.

$$\operatorname{MSE} = {1 \over n} \sum_{i=1}^n(y_i - \hat{f}(x_i))^2 $$

The MSE of an estimator $\hat{\theta}$ with respect to an unknown parameter $\theta$ is defined as

$$\operatorname{MSE}(\hat\theta)=E_\theta[ ({\hat \theta}-\theta )^{2} ]$$

MSE can be decomposed to a combination of bias and variance of the estimator

$${\displaystyle \operatorname {MSE} ({\hat \theta})=\operatorname {Var} _\theta({\hat \theta})+\operatorname {Bias} ({\hat \theta},\theta )^{2}}$$

Proof

Using the definition of variance

$\operatorname{Var}(X) = E(X^{2}) - (E(X))^{2}$ <br>
$E(X^{2})=\operatorname{Var}(X)+(E(X))^{2}$ 

By substituting $X$ with $\hat {\theta }-\theta$ it can be shown that

$\operatorname{MSE} ({\hat {\theta }})=\mathbb {E} [({\hat {\theta }}-\theta )^{2}]$<br>
$=\operatorname {Var} ({\hat {\theta }}-\theta )+(\mathbb {E} [{\hat {\theta }}-\theta ])^{2}$<br>
$=\operatorname {Var} ({\hat {\theta }})+\operatorname {Bias} ^{2}({\hat {\theta }})$

Variance is always positive and bias is squared. The selected estimator needs to minimize both variance and bias in order to minimize $\operatorname{MSE}$

We can scale MSE to be the same size as our data, this metric is called **Root Mean Sqaure Error** or RMSE

$$\operatorname{RMSE} = \sqrt{\operatorname{MSE}}$$

## **3.3 Regularization**

## **3.4 Model validation**

## **3.5 Hyper parameter tuning**
