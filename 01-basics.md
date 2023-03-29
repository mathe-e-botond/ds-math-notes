# 1. Statistics basics

This document will have basic equations needed to dereive statistical concepts.

## **1.1 Basics of probability**

Probability consitutes the basics of statistics. Probability theory is an application of measure theory and relies on set theory. 

### **1.1.1 Axioms of probability**

Probability is about possible worlds and probabiliatic assertions of how probable worlds are. **Sample space** is the set of all possible worlds. Possible worlds are *mutually exclusive* and *exhaustive*. A **random variables** is a measuremnt function that maps observations from a sample space to a measurable space, usually the real numbers $\mathbb{R}$. 

In the case of a random variable, for a fully specified **probability model** we can define a probability $P(A)$ for each possible outcome. 

Formally, let $(\Omega, F, P)$ be a measure space, called probability space for event $A$, sample space $\Omega$, event space $F$ and probability measure $P$

Probability can be described with a set of axioms named **Kolmogorov** axioms. 

Probability of each world can be defined as

$$P(A) \in \mathbb{R}, \\ 
\forall A \in F,\ \ 0 \le P(A) \tag{Axiom 1}$$

The total probability of all possible worlds is $1$

$$ P(\Omega) = 1 \tag{Axiom 2}$$

Set of worlds are called **events** or **propositions**. Probability of an event is the sum of the probability of the worlds which it contains. Formally it's the assumption of $\sigma$-additivity. For any countable sequence of disjoint sets $E_1...E_k$

$$ P(E_1 \cup ... \cup E_k) = P(E_1) + ... + P(E_k) \tag{Axiom 3} $$

### **1.1.4 Rules derived from axioms**

We can derive several consequences from the axioms

**Probability of empty set**

$$P(\phi) = 0$$

Proof

$E := \phi$<br>
$E \cup \phi = E$<br>
$P(E) + P(\phi) = P(E)$<br>
$P(\phi) = P(E) - P(E)$<br>
$P(\phi) = 0$

**Monotonicity**

$$A \subseteq B \implies P(A) \le P(B) $$

Proof

$A \subseteq B$<br>
$A \cup (B \setminus A) = B$<br>
$P(A) + P(B \setminus A) = P(B)$<br>
$P(A) \le P(B)$

**Complement rule**

$$ A^C = \Omega \setminus A \implies P(A^C) = 1 - P(A) $$

Proof

$A$ and $A^C$ are mutually exclusive and $A \cup A^C = \Omega$

$P(A \cup A^C) = P(A) + P(A^C)$<br>
$P(A) + P(A^C) = P(\Omega) = 1$<br>
$P(A^C) = 1 - P(A)$

**Numeric bound**

$$P(A) \le 1 \text{, for all A in event space}$$

Proof from the complement rool

$P(A^C) = 1 - P(A)$<br>
$P(A^C) \ge 0$, from the first axiom

**Inclusion-exclusion principle**

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Proof

$A$ and $B \setminus A$ are mutually exclusive

$P(A \cup B) = P(A \cup (B \setminus A))$<br>
$P(A \cup B) = P(A) + P(B \setminus A)$

Also $B \setminus A$ and $A \cap B$ are also exclusive with union B:

$(B \setminus A) \cup (A \cap B) = B$<br>
$P(B \setminus A) + P(A \cap B) = P(B)$

Adding both sides of the two results together

$P(A \cup B) + P(A \cap B) + P(B \setminus A) = P(A) + P(B) + P(B \setminus A)$, we can eliminate $P(B \setminus A)$

$P(A \cup B) + P(A \cap B) = P(A) + P(B)$<br>
$P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### **1.1.3 Conditional probabilities**

Probability of a propositions in the absence of any other information or **condition** is called an **unconditional probability**, **prior probability** or **prior**. In many cases there is already some **evidence**, in which case we can caculate the **conditional** or **posterior** probability. For propositions $A$ and $B$ conditional probabilities are defined as:

$$ P(A|B) = {P(A \cap B) \over P(B)} $$

which holds for $P(B) > 0$. We can also write this as the **product rule** or **chain rule**

$$ P(A \cap B) = P(A|B)P(B) $$

**Conditional probabilities act the same way as priors, because they satisfy the three axioms of probability**

1. $P(A|B) \ge 0$<br>
2. $P(B|B) = 1$<br>
3.  if $A_1, A_2, ..., A_k$ are mutually exclusive events, then<br>
$P(A_1 \cup ... \cup A_k | B) = P(A_1|B) + ... + P(A_k|B)$

Proof

1. $P(A \cap B) \ge 0, P(B) > 0 \implies {P(A \cap B) \over P(B)} \ge 0$<br>
2. $B \cap B = B$<br>
$P(B \cap B) = P(B)$<br>
$P(B|B) = {P(B \cap B) \over P(B)} = {P(B) \over P(B)} = 1$<br>
3. From set theory, for: $A_1...A_k$ mutually exclusive sets<br>
$(A_1 \cup ... \cup A_k) \cap B = (A_1 \cap B) \cup ... \cup (A_k \cap B)$<br>
$P((A_1 \cup ... \cup A_k) \cap B) = P((A_1 \cap B) \cup ... \cup (A_k \cap B))$<br>
$P((A_1 \cup ... \cup A_k) \cap B) = P(A_1 \cap B) + ... + P(A_k \cap B)$<br>
${P((A_1 \cup ... \cup A_k) \cap B) \over P(B)} = {P(A_1 \cap B) \over P(B)} + ... + {P(A_k \cap B) \over P(B)}$<br>
$P(A_1 \cup ... \cup A_k | B) = P(A_1 | B) + ... + P(A_k | B)$

### **1.1.4 Bayes rule**

The Bayes rule can be used to change the cause-effect probability to effect-cause or the otherway around. $P(\text{effect}|\text{cause})$ is the **casual** direction and $P(\text{cause}|\text{effect})$ is called the **diagnostic** direction

$$P(B | A) = {P(A | B) P(B) \over P(A)}$$

Proof using the product rule

$P(A \cap B) = P(A | B) P(B)$ and<br>
$P(A \cap B) = P(B | A) P(A)$ by making right side equal<br>
$P(B | A) P(A) = P(A | B) P(B)$<br>
$P(B | A) = {P(A | B) P(B) \over P(A)}$

Bayes rule can be conditioned on a background variable

$$P(B | A, e) = {P(A | B, e) P(B, e) \over P(A, e)}$$

instead of calculating $P(A, e)$ we can sometimes calculate the complement instead and normalizing it to become $1$

$$\text{using notation }\boldsymbol{P}(A) := \langle P(A), P(A^C) \rangle \\  
\boldsymbol{P}(B|A) = \alpha\boldsymbol{P}(B|A)\boldsymbol{P}(A) \\ = \alpha \langle P(B|A)P(A), P(B|A^C)P(A^C) \rangle $$

where $\alpha$ is the normalization constant to make entries in $\boldsymbol{P}$  sum up to $1$

## **1.2 Describing random variables** ##
If we enumerate all possible outcomes and their probabilities, we can construct a function that describes a random variable. This function is called **probability distribution**. 

### **1.2.1 Discrete probability distribution** ###

If the random variable outcome is discreete like a coint toss, the probability distribution function is also called **probability mass function**. 

$$p: \mathbb{R} \to [0, 1], \  p_X(x) = P(X = x)$$

Where values must be non negative and sum up to one as per the Kolmogorov axioms

$$p_X(x) \ge 0$$
and
$$\sum_x p_X(x) = 1$$

### **1.2.2 Continuous probability distribution** ###

In the case of **absolutely continuous probability distributions** the probability distribution is also called the **probability desnity function (PDF)**. Since the random varibale is continuous, the probability of taking of a specific value is 0. Instead we can describe the probability of a random variable taking a value from an interval 

$$Pr[a \le X \le b] = \int_a^bf(x)dx$$

Unlike the probability the density function can take up values bigger than $1$, but the integrsaste on the complete domain needs to be $1$

$$\int_{-\infty}^\infty f(x)dx = 1$$

### **1.2.3 Cumulative distribution function** ###

An alternative description with a funciton of a random variable is the **cumulative distribution function** (CDF) which in both discrete and continuous case is defined as the probability of the random variable taking a value bigger or equal to $x$.

$$F_X(x) = p(X <= x)$$

It has the following properties

$$\lim_{x \to - \infty} F(x) = 0\text{ and }\lim_{x \to  \infty} F(x) = 1$$

$$P(a < X <= b) = F_X(b) - F_X(a)$$

For discrete distribution the CDF is

$$F_X(x) = \sum_{k <= x} p(k)$$

For a continuous random variable

$$F_X(x) = \int_{-\infty}^x p(y)dy$$

## **1.3 Properties of probability distributions, populations and samples**

The purpose of statistics is to estimate properties of a population, given a sample. Properties of a population are for example what we call the moments of a random variable, defined as

1st moment: mean or expectation as central tendency<br>
2nd moment: variance<br>
3rd moment: skewness<br>
4th moment: kurtosis<br>

We can desfine each in terms of a population, a sample, discreete probability distribution or continuous probability distribution. 

### **1.3.1 Mean**

Population size N

$$\mu = {\sum x \over N}$$

Sample size $n$

$$\bar x = {\sum x \over n}$$

Discreete probability distribution

$$E[x] = \mu = \sum x p(x)$$

Continuous probability distribution

$$E[x] = \mu = \int_{-\infty}^\infty x f(x) dx$$

### **1.3.2 Variance**

Population size N

$$\sigma^2={\sum(x-\mu)^2 \over N}$$

Sample size $n$, for variance degrees of freedom is $n-1$ (for single observation variance is undefined)

$$s^2={\sum(x-\bar x)^2 \over n-1}$$

For probability distributions

$$\sigma^2 = Var(X) = E[(X - E(X))^2]$$

It can be shown that

$$E[(X - E(X))^2] = E[X^2] - (E[X])^2$$

Proof with both discrete and continuous random variables:
https://proofwiki.org/wiki/Variance_as_Expectation_of_Square_minus_Square_of_Expectation

$\sigma$ is called the standard deviation and is the square root of variance. For a probability distribution it's noted with $\operatorname{SD}(X)$

## **1.4 Multiple random variables** ##

We sometimes want to work with multiple random variables. 

### **1.4.1 Joint probability distribution** ###

The joint probability of two variables is noted by

$$P(A, B) = P(A \cap B)$$

The joint probability distribution is

$$f(x, y) = P(X = x, Y= y)$$

We can write it in terms of conditional distribution

$$P(X,Y) = P(X|Y)P(Y) = P(Y|X)P(X)\\
f(x, y) = P(X = x | Y= y) \cdot P(Y= y) = P(Y = y | X = x) \cdot P(X = x)$$

We can calulcate the individual probability distributions from the joint probability distribution, and it's called the **marginal probability distributions** (if we enumerate a discreet joint probability distribution in a table, we would calulcate the marginal distribution by summing up the rows and columns, making it the margin of the table as the last row and columns)

$$f_X(x) = \int f_{X,Y}(x,y)dy \\ f_Y(y) = \int f_{X,Y}(x,y)dx$$

Similarly to the probability distribution the joint cumulative distribution function

$$F_{X,Y}(x,y) = P(X \le x, Y \le y)$$

### **1.4.2 Independent and identically distributed random variables (i.i.d)** ###

Two variables are **independent** when the conditional probability is same as the pior

$$P(A|B) = P(A)$$

The joint probability distribution for two independent variables becomes

$$P(A,B) = P(A)P(B)$$

Independence can be stated with cumulative distribution functions

$$F_{X,Y}(x,y) = F_X(x)F_Y(y)\tag{i}$$

Two varaibales are identically distributed if their joint cumulative distribution function is equal

$$F_X(x) = F_Y(y)\tag{i.d}$$

Two variables are said to be **independent and identically distributed (i.i.d)** if both condition for independence (eq. (i)) and identically distributed (eq. (i.d.)) are both satisfied.

### **1.4.3 Covariance and correlation** ###

Similarity between two variables can be defined using correlation or covariance. 

For a random sample covariance is defined as

$$\operatorname{cov}(x,y) = \sigma_{xy} = {\sum(x - \bar x)(y - \bar y) \over n-1}$$

And correlation as

$$\operatorname{corr}(x,y) = \rho_{xy} = {\sigma_{xy} \over \sigma_x \sigma_y}$$

For a discreet probability distribution

$$\operatorname{cov}(X, Y) = \sum(X - E[X])(Y - E[Y])P(X,Y)$$

Correlation is

$$\operatorname{corr}(X, Y) = {\operatorname{cov}(X, Y) \over \operatorname{SD}(X)\operatorname{SD}(Y)}$$

## References

**Artificial Intelligence: A Modern Approach, Forth edition**
 Peter Norvig and Stuart J. Russell

https://en.wikipedia.org/wiki/Probability_axioms

https://en.wikipedia.org/wiki/Joint_probability_distribution

https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables

**Inclusion-exclusion principle** Marton Balaazs and Balint Toth, October 13, 2014

https://online.stat.psu.edu/stat414/lesson/4/4.2

https://machinelearningmastery.com/bayes-optimal-classifier/

https://medium.com/@luckecianomelo/the-ultimate-guide-for-linear-regression-theory-918fe1acb380

https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation