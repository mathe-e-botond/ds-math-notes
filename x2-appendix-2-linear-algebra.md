# Appendix II - Linear algebra

In this appendix we describe linear algebra, not from axioms, rather from intuitive, geometric understanding of vectors and matrices. 

Linear algebra is the study of vector spaces. For machine learning the most important vector space, is the **normed vector space**, which defines a norm, noted with $\|\cdot\|$, that simply means length.

The building blocks of vector spaces are real or integer numbers, called **scalars**, vectors and matrices.

We will build the intuition of vectors and matrices using examples in two dimensions, but all concepts equally apply for higher dimensions as well, visualizing concepts above two or three dimensions are extremely challenging and not too helpful.

## **Vectors**

There are many interpretations of vectors depending on the field of science. For vector spaces the simplest interpretation of an $n$ dimensional vector, is an $n$ dimensional arrow, having a tail always at the origin, and pointing to a coordinate defined by the $n$ scalars.

Vectors are usually noted with an arrow above a lowercase letter (e.g $\vec{v}$) or with an uppercase letter (e.g $V$). A vector can also be represented as an ordered list of it's $n$ components, usually written as a column:

$$V = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

$n$ is also called the dimensions of the vector. The set of all $n$ dimensional vectors of real numbers is noted with $\R^n$.

<p align="center">
<img src="./img/ii-vector.png" width="300">
<br><b>Figure II.1: </b>A two dimensional vector, having tail at origin (0, 0) and pointing to (a, b), where a and b are real numbers. Normed vector spaces define length, noted with a grayed grid as the unit steps on the space</p>

Figure II.1 shows an example of a two dimensional vector $\begin{pmatrix} a \\ b\end{pmatrix}$, where $a, b \in \R$. An intuition about vectors is that they represent movement, similar to the interpretation in physics.

Addition of two arrays is defined as pairwise addition of it's elements

$$U+V = \begin{pmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{pmatrix} + \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

In figure Figure II.2 we illustrate the geometric interpretation of the vector addition. If we continue our interpretation of vectors as movement, the sum of the two movement can be considered as continuing the second movement where the first ended. To represent this visually, we can move the tail of one vector to the arrow head of the other (this is the only exception when we move a vector's tail from origin). The resulting point of the second vector tip is the sum of the two vectors. 

<p align="center">
<img src="./img/ii-addition.png" width="300">
<br><b>Figure II.2: </b>Addition of two vectors U and V</p>

Multiplication of a vector with a scalar is defined as multiplying each element of the vector with the scalar:

$$c \cdot V = c \cdot \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot  v_n \end{pmatrix}$$

This will change the scale (or norm) of the vector but not the direction of the line the vector is on. We can say the scalar scales the vector, hance the name. A negative value will flip the vector to opposite direction (on the same line), a value between $(1, +\infty)$ will enlarge it, a scalar of $1$ is a unit operation and a value in the interval $(0, 1)$ will decrease the scale of the vector.

Scaling and addition of multiple vectors is called the **linear combination** of vectors.

$$V = c_1 V_1 + ... + c_k V_k$$

## **Basis and span**

A set of $n$ vectors, each with $n$ dimensions, noted with $B$, with elements $\hat{b}_1, ..., \hat{b}_n$ are **linearly independent** if none of the vectors can be expressed as a linear combination of the other vectors using non zero scalars. Which means if $\hat{b}_1, ..., \hat{b}_n$ are linearly independent, the expression

$$c_1\hat{b}_1 + ... + c_n \hat{b}_n = 0$$

can be true only if scalars $c_1, ..., c_n$ are all $0$. We could visualize this by imagining that each vector points toward a different dimension.

Given a vector space $S$ with $n$ dimensions, we define the **basis** of $S$ as a set of vectors $B$ with $n$ linearly independent vectors $\hat{b}_1, ..., \hat{b}_n$. Any vector in the vector space can be expressed as a linear combination using the basis vectors: $\forall\ V, V \in S$, we can choose $n$ scalars $c_1, ..., c_n$ such that

$$V = c_1\hat{b}_1 + ... + c_n\hat{b}_n$$

A very common basis vector, also called the **standard basis vector** is a vector where each element is $0$ except for a single dimension it expands with a value of $1$.

In Figure II.3 we have a two dimensional vector space (a plane) noted by a grey grid. The standard basis vectors are $\hat{i} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ and $\hat{j} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$, making the standard basis $B = \{\hat{i}, \hat{j}\}$. 

<p align="center">
<img src="./img/ii-span-and-base.png" width="300">
<br><b>Figure II.3: </b>Bases of a vector space</p>

We can express $V = \begin{pmatrix} a \\ b \end{pmatrix}$ as:

$$V = a \hat{i} + b \hat{j}$$

We scale each base vector and add the result to get any vector in the vector space. We can imagine that the base of a vector space are a set of vectors, that fills out the space trough scaling and additions (i.e linear combinations). T

he vector $V$ in Figure II.3 and any other vector of this vector space, can be also expressed trough a linear combination of $\hat{u}$ and $\hat{v}$. We can say that the set of vectors $\{\hat{u}$, $\hat{v}\}$ is also a basis for this vector space, but the coordinates of $V$ would be $\begin{pmatrix} 1 \\ 1 \end{pmatrix}$ instead of $\begin{pmatrix} a \\ b \end{pmatrix}$, because $V = 1 \cdot \hat{u} + 1 \cdot \hat{v}$.

**Span** of a set of vectors with same dimensions $n$ is the vector space that can be covered trough the linear combination of the vectors. As we have seen, if all the vectors are linearly independent, they will constitute the basis of an $n$ dimensional vector space. If only $p$ vectors are linearly independent and the rest of $n - p$ vectors can be expressed as linear combination of the other vectors, the span will be a $p$ dimensional vector space. Two vectors $\hat{a}$ and $\hat{b}$ in three dimensions might span a plane if they are linearly independent, a line if they are linearly dependent ($\hat{a} = c \cdot \hat{b}$ for some scalar $c$) or the span might be the point of origin if $\hat{a} = \hat{b} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$

## **Matrices**

The most important operation in linear algebra are linear transformations. Linear transformation is a way to change all vectors of a vector space according to a linear operator. Transformations on a vector space can be non linear as well, where we can bend or apply some wave to the space, but linear transformation have two properties

* Linear transformation does not change the origin of space
* Any line in the original space retains it's shape as line in the transformed shape (there is no bending) 

With these constraints the mathematics of linear transformations are more simple, and faster to compute, but still very useful. To apply linear transformation matrices are used. A matrix $M_{n,m}$ is a two dimensional structure, noted as a grid

$$M_{n,m} =  \begin{pmatrix}
  x_{1,1} & x_{1,2} & \cdots & x_{1,p} \\
  x_{2,1} & x_{2,2} & \cdots & x_{2,p} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  x_{n,1} & x_{n,2} & \cdots & x_{n,p} 
 \end{pmatrix}$$

Where $x_{1...n,1...m}$ are real numbers. The set of all matrices of size $n \cdot m$ containing real numbers is noted with $\R^{n \cdot m}$.

### **Square matrix**

To understand the structure of transformation matrix, let's start with a certain type of matrix, where the number of rows and columns are equal, called **square matrix**, which we can note $M_{n, n}$. Such a matrix describes the linear transformation in an $n$ dimensional vector space with both input and output vectors having $n$ dimensions. 

Linear transformation is interpreted as transforming the basis of a vector space to another basis. To transform the $k$ th dimension, we can take the $k$ th standard basis vector, where each element is $0$ except for the $k$ th element, which is 1, and calculate where it would be in the transformed space.

$$\vec i_k = \begin{pmatrix} 0 \\ \vdots \\ 0 \\ 1 \\ 0 \\  \vdots \\ 0 \end{pmatrix} \rightarrow \vec b_k = \begin{pmatrix} b_{k,1} \\ \vdots \\ b_{k,k-1} \\ b_{k,k} \\ b_{k,k+1} \\ \vdots \\ b_{k,n} \end{pmatrix} $$

We can imagine that each column of a matrix is a vector, the matrix $M_{n, n}$ has $n$ vectors, each of $n$ dimensions. The transformation matrix is simply the matrix constructed by conbining all the basis vectors we would get by transforming each of the standard basis vectors.

$$I_{n, n} = \begin{pmatrix}
  1 & 0 & \cdots & 0 \\
  0 & 1 & \cdots & 0 \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  0 & 0 & \cdots & 1 
 \end{pmatrix} \rightarrow M_{m,n} = \begin{pmatrix}
  b_{1,1} & b_{1,2} & \cdots & b_{1,n} \\
  b_{2,1} & b_{2,2} & \cdots & b_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  b_{n,1} & b_{n,2} & \cdots & b_{n,n} 
 \end{pmatrix} \tag{II.1}$$

To calculate the linear transformation of a vector we use the transformation matrix and we apply an operation called matrix-vector multiplication defined as:

$$V_t = M_{n,m} \cdot V \\ = \begin{pmatrix}
  x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\
  x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  x_{n,1} & x_{n,2} & \cdots & x_{n,n} 
 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} \\ = v_1\begin{pmatrix}
  x_{1,1} \\ x_{2,1} \\ \vdots \\ x_{n,1} \end{pmatrix} + v_2\begin{pmatrix}
  x_{1,2} \\ x_{2,2} \\ \vdots \\ x_{n,2} \end{pmatrix} + ... + v_n\begin{pmatrix}
  x_{1,n} \\ x_{2,n} \\ \vdots \\ x_{n,n} \end{pmatrix} \\ = \begin{pmatrix}
  x_{1,1} v_1 + x_{1,2} v_2 + \cdots + x_{1,n} v_n  \\
  x_{2,1} v_1 + x_{2,2} v_2 + \cdots + x_{2,n} v_n  \\
  \vdots \\
  x_{n,1} v_1 + x_{n,2} v_2 + \cdots + x_{n,n} v_n  \\ 
 \end{pmatrix} \tag{II.2}$$

The resulting vector $V_t$ is also $n$ dimensional. The matrix $I_{n, n}$ in Figure II.1 is called the identity matrix. If we use this matrix as a transformation operator for a vector $V$, it would result in the same vector $I_{n,n} \cdot V = V$.

<p align="center">
<img src="./img/ii-matrix.png" width="300">
<br><b>Figure II.4: </b>Vector space transformation</p>

In Figure II.4 we can see the visual representation of a two dimensional vector space transformation. The standard base of the original space is $\{\vec i, \vec j\}$ and it's shown as a gray grid. The transformed space has standard base $\{\vec u, \vec v\}$, these vectors correspond to $\vec i$ and $\vec j$ in the original vector space. We can imagine that the space is stretched so that $\vec i$ and $\vec j$ are moved to the place of $\vec u$ and $\vec v$.

$$\vec i \rightarrow \vec u, \vec j \rightarrow \vec v$$

In Figure II.4 the transformed space can be seen as an elongated diagonally skewed grid of black lines. This transformation is described by a $2 \cdot 2$ matrix, having first column the elements of vector $u$ and second columns with elements of vector $v$.

$$M_{2,2} = \begin{pmatrix}
  u_{1} & v_{1} \\
  u_{2} & v_{2} 
 \end{pmatrix}$$

If we apply the matrix-vector multiplication in the Formula II.2 to the matrix $M_{2,2}$ and $\vec i$, we get $\vec u$:

$$M_{2,2} \cdot \vec i = \begin{pmatrix}
  u_{1} & v_{1} \\
  u_{2} & v_{2} 
 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = 
 \begin{pmatrix} 1 \cdot u_{1} + 0 \cdot v_{1} \\ 1 \cdot u_{2} + 0 \cdot v_{2} \end{pmatrix} = \vec u$$

Similarly we can show that $\vec j \rightarrow \vec v$. $M_{2,2}$ similarly transforms all vectors in the space. 

<p align="center">
<img src="./img/ii-matrix-2.png" width="300">
<br><b>Figure II.5: </b>The transformation applied to a vector</p>

Figure II.5 shows the effect of applying the linear transformation of our example matrix $M_{2,2}$ on vector $\vec t$, resulting in vector $\vec t'$. We can see that the vector space was not only elongated but also flipped around the axis of vector $\vec z$

### **Special transformation in 2D and 3D**

### **Nonsquare matrices**

### **Matrix multiplication**

## **Determinant**


## **Inverse matrices**


## **Eigenvalues and eigenvectors**


## **Singular value decomposition**
