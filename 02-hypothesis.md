# 2. Hypothesis testing

Hypothsis testing is the process to confirm a test metric on a data set. The general process is to

1. State a **null hypothesis** $H_0$ which is the contradiction of the **alternative hypothesis** we want to verify, sometimes noted with $H_1$
2. Use a **test statistic** to calculate the probability of an observation given the null hypothesis. This probability is the **p-value**
3. Compare the p-value to a target $\alpha$ **significance level**

We say a null hypothesis is one tailed if

$$H_0: \mu = \mu_0,\ H_1 : \mu > \mu_0 \text{ or } H_1 : \mu < \mu_0$$

a two-tailed test is

$$H_0: \mu = \mu_0,\ H_1 : \mu \ne \mu_0$$

for some metric $\mu$

## **2.1. Z-test and t-test**

## **2.2. Anova**

Anova is used to verify means of multiple populations. If we apply Z-test multiple times, the error accumulates.

The Anova Hypothesis for $p$ groups:

$$
\begin{aligned}
&H_0: \mu_1 = \mu_2 = ... = \mu_p,\  \\
&H_1: \mu_i \ne \mu_j,\ \forall i, j \in \{1, ...\ , p\}
\end{aligned}
$$

