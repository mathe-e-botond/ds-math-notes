# 2. Regression

Linear models are a set of supervisied statistical learning techniques, used to approximate an unknown function based on observations. 

## 2.1 Estimators

We can model an **independent** variable with the **dependent** variables of some observations. We assume the following relationship

$$Y = f(X) + \epsilon \tag{2.1}$$

We define estimate in the form

$$\hat{Y} = \hat{f}(X) \tag{2.2}$$

### **2.1.1 The error term**

$\epsilon$ is the error term and can be decomposed:

$$E[(Y-\hat{Y})^2] = \underbrace{[f(X)-\hat{f}(X)]^2}_\text{Reducible} + \underbrace{\text{Var}(\epsilon)}_\text{Irreducible}$$

Proof:

Using (2.1) and (2.2)<br>
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

We can opimize our estimate to minimize reducible error but irreducible error is also unknown and our model might overfit by inclusing it to the estimate.

## 2.1.2 Model fit and bias-variance trade-off ##

Model fit can be measured with various metrics, the most popular being *mean squared error* or MSE

$$MSE = {1 \over n} \sum_{i=1}^n(y_i - \hat{f}(x_i))^2 $$

The MSE of an estimator $\hat{\theta}$ with respect to an unknown parameter $\theta$ is defined as

$$MSE(\hat\theta)=E_\theta[ ({\hat \theta}-\theta )^{2} ]$$

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

## **2.2 Linear regression**

For $X_1...X_p$ predictor variables and $\epsilon$ irreducible error term, linear regression model has the form

$$y = \beta _0 + \beta _1 x_1 + ... + \beta _p x_p + \epsilon$$

The regression coefficinets $\beta_1...\beta_p$ are unknown, an estimate takes the form

$$\hat y = \hat \beta _0 + \hat \beta _1 x_1 + ... +  \hat \beta _p x_p$$

Parameters can be estimated using our training data set and the sum of squared residual loss function

$$RSS = \sum_{i=1}^n(y_i - \hat y)^2$$

To assess quality of fit, the **residual standard error** $RSE$ is an estimate of standard deviation of $\epsilon$

$$RSE = \sqrt{RSS \over n - p - 1}$$

## References

**An Introduction to Statistical Learning, with applications in R, Second Edition**, Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani

https://stats.stackexchange.com/questions/191113/proof-for-irreducible-error-statement-in-islr-page-19

https://en.wikipedia.org/wiki/Mean_squared_error#Proof_of_variance_and_bias_relationship