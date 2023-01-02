# 2. Linear models

Linear models are a set of supervisied statistical learning techniques, used to approximate an unknown function based on observations. 

## 2.1 Estimators

We can model an **independent** variable with the **dependent** variables of some observations. We assume the following relationship

$$Y = f(X) + \epsilon \tag{2.1}$$

We define estimate in the form

$$\hat{Y} = \hat{f}(X) \tag{2.2}$$

### **2.1.1 The error term**

$\epsilon$ is the error term and can be decomposed using (2.1) and (2.2):

$$[(Y-\hat{Y})^2] = \underbrace{[f(X)-\hat{f}(X)]^2}_\text{Reducible} + \underbrace{\text{Var}(\epsilon)}_\text{Irreducible}$$

Proof:

$E[(Y-\hat{Y})^2]$<br>
$=E[(f(X)+\epsilon-\hat{f}(X))^2]$<br>
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

## 2.2 ##

## References

**An Introduction to Statistical Learning, with applications in R, Second Edition**, Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani

https://stats.stackexchange.com/questions/191113/proof-for-irreducible-error-statement-in-islr-page-19

