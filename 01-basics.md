# 1. Statistics basics

This document will have basic equations needed to dereive statistical concepts

## **1.1 Probability**

Probability consitutes the basics of statistics. Probability theory is an application of measure theory and relies on set theory. 

### **1.1.1 Axioms of probability**

Probability is about possible worlds and probabiliatic assertions of how probable worlds are. **Sample space** is the set of all possible worlds. Possible worlds are *mutually exclusive* and *exhaustive*. In the case of discrete countable set of worlds, for a fully specified **probability model** we define a probability $P(A)$ for each world. 

Formally, let $(\Omega, F, P)$ be a measure space, called probability space for event $A$, sample space $\Omega$, event space $F$ and probability measure $P$

Probability can be described with a set of axioms named **Kolmogorov** axioms. 

Probability of each world is a non negative real number bigger than $0$

$$P(A) \in \mathbb{R}, \\ 
\forall A \in F,\ \ 0 \le P(A) \tag{Axiom 1}$$

The total probability of all possible worlds is $1$

$$ P(\Omega) = 1 \tag{Axiom 2}$$

Set of worlds are called **events** or **propositions**. Probability of an event is the sum of the probability of the worlds which it contains. Formally it's the assumption of $\sigma$-additivity. For any countable sequence of disjoint sets $E_1...E_k$

$$ P(E_1 \cup ... \cup E_k) = P(E_1) + ... + P(E_k) \tag{Axiom 3} $$

### **1.1.2 Rules derived from axioms**

We can derive several consequences from the axioms

**Probability of empty set**

$$P(\phi) = 0$$

Proof

$E := \phi \\
E \cup \phi = E \\
P(E) + P(\phi) = P(E) \\
P(\phi) = P(E) - P(E) \\
P(\phi) = 0$

**Monotonicity**

$$A \subseteq B \implies P(A) \le P(B) $$

Proof

$A \subseteq B \\
A \cup (B \setminus A) = B\\
P(A) + P(B \setminus A) = P(B) \\
P(A) \le P(B)$

**Complement rule**

$$ A^C = \Omega \setminus A \implies P(A^C) = 1 - P(A) $$

Proof

$A$ and $A^C$ are mutually exclusive and $A \cup A^C = \Omega$

$P(A \cup A^C) = P(A) + P(A^C) \\
P(A) + P(A^C) = P(\Omega) = 1 \\
P(A^C) = 1 - P(A)$

**Numeric bound**

$$P(A) \le 1 \text{, for all A in event space}$$

Proof from the complement rool

$P(A^C) = 1 - P(A) \\
P(A^C) \ge 0 \text{, from the first axiom} \\
$

**Inclusion-exclusion principle**

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Proof

$A$ and $B \setminus A$ are mutually exclusive

$P(A \cup B) = P(A \cup (B \setminus A)) \\
P(A \cup B) = P(A) + P(B \setminus A)$

Also $B \setminus A$ and $A \cap B$ are also exclusive with union B:

$(B \setminus A) \cup (A \cap B) = B \\
P(B \setminus A) + P(A \cap B) = P(B)$

Adding both sides of the two results together

$P(A \cup B) + P(A \cap B) + P(B \setminus A) = P(A) + P(B) + P(B \setminus A)$, we can eliminate $P(B \setminus A)$

$P(A \cup B) + P(A \cap B) = P(A) + P(B) \\
P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### **1.1.3 Conditional probabilities**

Probability of a propositions in the absence of any other information or **condition** is called an **unconditional probability**, **prior probability** or **prior**. In many cases there is already some **evidence**, in which case we can caculate the **conditional** or **posterior** probability. For propositions $A$ and $B$ conditional probabilities are defined as:

$$ P(A|B) = {P(A \cap B) \over P(B)} $$

which holds for $P(B) > 0$. We can also write this as the **product rule** or **chain rule**

$$ P(A \cap B) = P(A|B)P(B) $$

**Conditional probabilities act the same way as priors, because they satisfy the three axioms of probability**

$\text{1. } P(A|B) \ge 0 \\
\text{2. } P(B|B) = 1 \\
\text{3.  if} A_1, A_2, ..., A_k\text{ are mutually exclusive events, then} \\
P(A_1 \cup ... \cup A_k | B) = P(A_1|B) + ... + P(A_k|B)
$

Proof

$\text{1. } P(A \cap B) \ge 0, P(B) > 0 \implies {P(A \cap B) \over P(B)} \ge 0 \\
\text{2. } B \cap B = B \\
P(B \cap B) = P(B) \\
P(B|B) = {P(B \cap B) \over P(B)} = {P(B) \over P(B)} = 1 \\
\text{3. From set theory, for: } A_1...A_k \text{ mutually exclusive sets} \\
(A_1 \cup ... \cup A_k) \cap B = (A_1 \cap B) \cup ... \cup (A_k \cap B) \\
P((A_1 \cup ... \cup A_k) \cap B) = P((A_1 \cap B) \cup ... \cup (A_k \cap B)) \\ 
P((A_1 \cup ... \cup A_k) \cap B) = P(A_1 \cap B) + ... + P(A_k \cap B) \\ 
{P((A_1 \cup ... \cup A_k) \cap B) \over P(B)} = {P(A_1 \cap B) \over P(B)} + ... + {P(A_k \cap B) \over P(B)} \\
P(A_1 \cup ... \cup A_k | B) = P(A_1 | B) + ... + P(A_k | B)$

### **1.1.4 Bayes rule**

The Bayes rule can be used to change the cause-effect probability to effect-cause or the otherway around. $P(\text{effect}|\text{cause})$ is the **casual** direction and $P(\text{cause}|\text{effect})$ is called the **diagnostic** direction

$$P(B | A) = {P(A | B) P(B) \over P(A)}$$

Proof using the product rule

$ P(A \cap B) = P(A | B) P(B)  \text{ and} \\
P(A \cap B) = P(B | A) P(A)  \text{ by making right side equal} \\
P(B | A) P(A) = P(A | B) P(B) \\
P(B | A) = {P(A | B) P(B) \over P(A)}
$

Bayes rule can be conditioned on a background variable

$$P(B | A, e) = {P(A | B, e) P(B, e) \over P(A, e)}$$


instead of calculating $P(A, e)$ we can sometimes calculate the complement instead and normalizing it to become $1$

$$\text{using notation }\boldsymbol{P}(A) := \langle P(A), P(A^C) \rangle \\  
\boldsymbol{P}(B|A) = \alpha\boldsymbol{P}(B|A)\boldsymbol{P}(A) \\ = \alpha \langle P(B|A)P(A), P(B|A^C)P(A^C) \rangle $$

where $\alpha$ is the normalization constant to make entries in $\boldsymbol{P}$  sum up to $1$


## References

**Artificial Intelligence: A Modern Approach, Forth edition**
 Peter Norvig and Stuart J. Russell

https://en.wikipedia.org/wiki/Probability_axioms

**Inclusion-exclusion principle** Marton Balaazs and Balint Toth, October 13, 2014

https://online.stat.psu.edu/stat414/lesson/4/4.2
