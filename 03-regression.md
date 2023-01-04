# 3. Regression

Linear models are a set of supervisied statistical learning techniques, used to approximate an unknown function based on observations. 

## **3.1 Estimators**

We can model an **independent** variable with the **dependent** variables of some observations. We assume the following relationship

$$Y = f(X) + \epsilon \tag{3.1}$$

We define estimate in the form

$$\hat{Y} = \hat{f}(X) \tag{3.2}$$

### **3.1.1 The error term**

$\epsilon$ is the error term and can be decomposed:

$$E[(Y-\hat{Y})^2] = \underbrace{[f(X)-\hat{f}(X)]^2}_\text{Reducible} + \underbrace{\text{Var}(\epsilon)}_\text{Irreducible}$$

Proof:

Using (3.1) and (3.2)<br>
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

## 3.1.2 Model fit and bias-variance trade-off ##

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

## **3.2 Linear regression**

Linear regression tries to map continuous or categorical variables to a continuous independent variable. For $X_1...X_p$ predictor variables and $\epsilon$ irreducible error term, linear regression model has the form

$$y = \beta _0 + \beta _1 x_1 + ... + \beta _p x_p + \epsilon$$

The regression coefficinets $\beta_1...\beta_p$ are unknown, an estimate takes the form

$$\hat y = \hat \beta _0 + \hat \beta _1 x_1 + ... +  \hat \beta _p x_p$$

Parameters can be estimated using our training data set and the sum of squared residual loss function

$$RSS = \sum_{i=1}^n(y_i - \hat y)^2$$

### **3.2.1 Assess quality of fit**

To assess quality of fit, the **residual standard error** $RSE$ is an estimate of standard deviation of $\epsilon$

$$RSE = \sqrt{RSS \over n - p - 1}$$

$RSE$ is on same scale as $y$, $R^2$ is another measure of fit on the scale between $0$ and $1$

$$
\begin{aligned}
R^2 &= {\operatorname{variance\ explained} \over \operatorname{total\ variance}} = {Var(\operatorname{mean}) - Var(\operatorname{fit}) \over Var(\operatorname{mean})}
\\
&= {{TSS \over n} - {RSS \over n} \over {TSS \over n}} = {TSS - RSS \over TSS} = 1 - {RSS \over TSS}
\end{aligned}
$$

where $TSS$ is **total sum of squares**: $TSS = \sum(y_i - \bar y)^2$. $\bar y$ is the average of $y$. $RSS$ is the amount of variability unexplained after regression, $TSS$ is the total variability. Some text books use this notation

$$R^2 = {SS(\operatorname{mean}) - SS(\operatorname{fit}) \over SS(\operatorname{mean})} $$

In simple liner regression setting ($y = \beta _0 + \beta _1 x$), $R^2$ is same as $Cor(X, Y)^2$

If we have $p+1$ observations $R^2$ will **always be $1$** because we are fitting a $p$ dimensional plane on $p+1$ points so there is always a perfect fit. Such a fit hovewer has very litttle confidence. To calculate p-value for $R^2$ we use the $F$ value 

$$F = {\operatorname{variance\ explained} \over \operatorname{variance\ not\ explained}} = {SS(\operatorname{mean}) - SS(\operatorname{fit}) / (p_{\operatorname{fit}} - p_{\operatorname{mean}}) \over SS(\operatorname{fit}) / (n - p_{\operatorname{fit}})}$$

Where in case of linear regression $p_{\operatorname{mean}}$ is $1$

We can either simulate lots of F values by sampling our data, calculating the fitted line and F score and finally calculating the F score for the whole dataset and finding percentiles of more extreme values from our simulations, or we can use the *F*-distribution  

### **3.2.3 Categorical predictors**

Categorical predictors can be added trough dummy encoding. In this case the equation will be

$$y = \beta_0 + \sum_{c \in C}\beta_cx_c$$

WHere $C$ contains all categorical values except one which will act as baseline (and be part of intercept). If all dummies are included, it will cause multi colliniarity because the last dummy explains the others.

### **3.2.4 Extensions to linear model**

We can remove additive assumptions by creating custom predictors combining other predictors called **interactions** or adding **polinomial terms** . E.g.

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_3x_1x_2 + \beta_4x_2^2 + \beta_5x_2^3$$

When we include interactions or polinomial terms we should always include the base predictors called **main effects**. Categorical variables only contribute to intercept, to add slope effect as well, needs to be added to interaction

### **3.2.2 Cosniderations of linear regression**

1. **Linear relationship** between predictors and dependent variable. Can be verified using residual plots ($e_i = y_i - \hat y_i$ vs $x_i$ or in the case of multiple regression $y_i$). In case of non linearity polinomyal terms can be used

2. **Error terms are uncorrelated**, an error term $\epsilon_i$ provides no information about $\epsilon_{i+1}$, like sign or distance, which is the case for ezample for time series analysis

3. **Constant variance of error terms**: if error terms increase with dependent variable, it's called **heteroscandecity** and can be seen on the residual plot as a funnel shape. In the case of this issue, we can transofrm the response using a concave function ($\sqrt{y}$ or $log(y)$). Another option might be to fit using **weighted least squares**

4. **Outliers** can be identified from residual plot, or we can plot **studentized residuals** $$\bigg|{\epsilon_i \over SE}\bigg| > 3$$ <br>
Outliers might indicate incorrect data input in which case can be simply removed or issues with model like missing predictor variable

5. **High levarege points** are observations who have unusual predictor $x_i$ values and might easily influence the regression. A so called **levarage statistic** can be calculated to quantify levarage, more so for multiple predictors

6. **Multicolliniarity** happens if there is corralation between predictor variables. Pairwise correlation can be detected by plotting the correlation matrix of the predictors. Can be quantified trough the **variance inflation factor** (VIF) $$\operatorname{VIF}(\beta_j) = {1 \over 1 - R_{X_jX-j}^2}$$
where $R_{X_jX-j}^2$ is the $R^2$ of a regression of $X_j$ to the other predictors. Minimum value for VIF is 1, a value above 5 or 10 indicated multicolliniarity. 
In case of multicolliniarity we can remove one of the predictors or combine multiple predictors into one.

Interactions and polinomial terms can cause multicolliniarity for non centered predictors, centering solves this issue, see [here](https://stats.stackexchange.com/questions/60476/collinearity-diagnostics-problematic-only-when-the-interaction-term-is-included).

## **3.3 K-Nearest Neighbour regression**

KNN is a non parametric estimator, and so does not make assumptions about the form of $f(X)$. On the other hand, does not support inference (explaining predictor relationships). To perform KNN regression, we find the K nearest neighbours of $x_0$ noted with $\Nu_0$, and we calculate the average of training responses

$$f(x_0) = {1 \over K} \sum_{x_i \in |Nu_0}y_i$$

A parametric approach usually ouperforms the non parametric one, becuase the non parametric can have an increase in variance without reducing bias. With large number of predictors, the **curse of dimensionality** reduces the number of neighbours that can be used. In some cases KNN might perform better, but model explainability and the presence of p-values is also an advantage of linear regression.

## References

**An Introduction to Statistical Learning, with applications in R, Second Edition**, Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani

https://stats.stackexchange.com/questions/191113/proof-for-irreducible-error-statement-in-islr-page-19

https://en.wikipedia.org/wiki/Mean_squared_error#Proof_of_variance_and_bias_relationship