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

A special vector, of size $0$, starting and ending in the origin has all elements as $0$.

$$O = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}$$

Figure II.1 shows an example of a two dimensional vector $\begin{pmatrix} a \\ b\end{pmatrix}$, where $a, b \in \R$. An intuition about vectors is that they represent movement, similar to the interpretation in physics.

<p align="center">
<img src="./img/ii-vector.png" width="300">
<br><b>Figure II.1: </b>A two dimensional vector, having tail at origin (0, 0) and pointing to (a, b), where a and b are real numbers. Normed vector spaces define length, noted with a grayed grid as the unit steps on the space</p>

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

### **Norm of vector**

In normed vector spaces each vector has a norm, which corresponds to length in every day language. Norm is different from the dimensions. We define norm as

$$\|V\| = \sqrt {v_1^2 + v_2^2 + ... + v_n^2}$$

### **Dot product and orthogonal vectors**

The dot product of two vectors of same dimensions $n$ is defined as the sum of pairwise products of their elements.

$$U \cdot V = u_1 v_1 + u_2 v_2 + ... + u_n v_n$$

The dot product is defined to result in $0$ if $U$ and $V$ make a $90 \degree$ angle, i.e $U$ and $V$ are perpendicular vectors. Another word with same meaning is **orthogonal** vector.

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

$$V_t = M_{n,n} \cdot V \\ = \begin{pmatrix}
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

$$M_{uv} = \begin{pmatrix}
  u_{1} & v_{1} \\
  u_{2} & v_{2} 
 \end{pmatrix}$$

If we apply the matrix-vector multiplication in the Formula II.2 to the matrix $M_{2,2}$ and $\vec i$, we get $\vec u$:

$$M_{uv} \cdot \vec i = \begin{pmatrix}
  u_{1} & v_{1} \\
  u_{2} & v_{2} 
 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = 
 \begin{pmatrix} 1 \cdot u_{1} + 0 \cdot v_{1} \\ 1 \cdot u_{2} + 0 \cdot v_{2} \end{pmatrix} = \vec u$$

Similarly we can show that $\vec j \rightarrow \vec v$. $M_{2,2}$ similarly transforms all vectors in the space. 

<p align="center">
<img src="./img/ii-matrix-2.png" width="300">
<br><b>Figure II.5: </b>The transformation applied to a vector</p>

Figure II.5 shows the effect of applying the linear transformation of our example matrix $M_{uv}$ on vector $\vec t$, resulting in vector $\vec t'$. We can see that the vector space was not only elongated but also flipped around the axis of vector $\vec z$

## **Determinant**

A transformation might expand or condense the vector space. The **determinant** of a matrix measures the rate of expansion applied to a vector space by the matrix. 

The standard basis vectors (what we noted with $\vec i_k, k = 1..n$) define a $1 \times 1 \times ... \times 1$ hypercube, where each standard basis vector would be an edge of the cube around the corner where the origin $O$ is. If we apply the linear transformation defined by the matrix to this cube, we get a shape called **parallelotope** (parallelogram in 2D, parallelepiped in 3D). The determinant is the area of the resulting shape after applying the matrix transformation. The determinant will be the same if we apply the transformation to any $n$ dimensional hypercube in the vector space. Any shape that can be approximated with such hypercubes will also scale with the same factor. The determinant is only defined for square matrices.

The determinant can have the following meaning depending on it's value:
* Determinant of $1$ means there is no change in the area of hypercube when applying the matrix transformation. This is the case for example for rotation and **sheer** operation.
* A determinant of $0$ of a matrix means the area of the hypercube is scaled down to $0$ with the matrix transformation. This is called **projection**, we reduce one or more dimension by projecting all points of our hyperspace to a lower dimension hyperplane or all the way down the point of origin. This also means that some or all vectors from the matrix columns are linearly dependent, based on how many dimensions we reduce. 

    The number of linearly independent columns (taken as vectors) in a square matrix is called the **rank** of the matrix. Rank is also the number of dimensions of the hyperplane resulting when applying the matrix transformation to our vector space. If the rank is equal to the number of columns, the determinant is not $0$.

    In the case of a projection a line or a hyperplane or the entire hyperspace will be projected to the point of origin $O$. The line, plane or hyperspace that ends up as the origin after the transformation is called the **kernel** of the matrix.  
* The determinant will be negative for one or any odd number of flips in the hyperspace. An even number of flips restores the space to it's original "side", the same transformation can be achieved trough rotation. The absolute value of a negative determinant will tell the factor the space is being scaled.

The computation itself for the determinant is more complex for each added dimension, but here we will explore the two dimensional case and it's computation.

<p align="center">
<img src="./img/ii-determinant.png" width="500">
<br><b>Figure II.6: </b>Determinant of a matrix</p>

In Figure II.6 we can see two vectors of $\vec a = \begin{pmatrix} a_1 \\ a_2 \end{pmatrix}$ and $\vec b = \begin{pmatrix} b_1 \\ b_2 \end{pmatrix}$. They can form the transformation matrix $M_{ab} =  \begin{pmatrix}
  a_{1} & b_{1} \\
  a_{2} & b_{2} 
 \end{pmatrix}$. If we apply $M$ as the transformation matrix to the basis $\{\vec i, \vec j\}$ we get the new basis $\{\vec a, \vec b\}$. The area of the $1 \times 1$ square resting on the $\{\vec i, \vec j\}$ becomes the shaded area resting on $\{\vec a, \vec b\}$. The determinant of the matrix $M$ is the resulting area:

 $$\det(M_{ab}) = \det\begin{pmatrix}
  a_{1} & b_{1} \\
  a_{2} & b_{2} 
 \end{pmatrix} = a_1b_2 - b_1a_2$$

This can be derived looking at the right side of Figure II.6:

$\det(M_{ab}) = (a_1 + b_1)(a_2 + b_2) - 2 {a_1a_2 \over 2}  - 2 {b_1b_2 \over 2} - 2 a_2 b_1$ <br>
$= a_1a_2 + a_1b_2 + a_2b_1 + b_1b_2 - a_1a_2 - b_1b_2 - 2 a_2 b_1$ <br>
$= a_1b_2 - a_2b_1$

If we create another matrix with the columns flipped $M_{ba} = \begin{pmatrix}
  b_{1} & a_{1} \\
  b_{2} & a_{2} 
 \end{pmatrix}$ this would not only scale the vector space but also flip as we have seen in the case of $M_{uv}$ flipping the space around $\vec z$, on Figure II.5. In this case the determinant is negative, but absolute value remains the same: 
 
 $$\det(M_{ab}) = -\det(M_{ba})$$

## **Special transformations with square matrices**

Depending on the type of transformation we can define some special matrices

* **Identity matrix** $I_{n,n}$ has values of $1$ on it's diagonal and $0$ on every other, off-diagonal element. We have seen that this matrix preserves all vectors without any transformation.

* **Scalar matrix** has values of $k$ on it's diagonal and $0$ on every other, off-diagonal element. We can express a scalar matrix as $kI_{n,n}$. What this matrix does is scales the vector space by a scale of $k$. For values $k > 1$ the vector space is expanded, for values in the interval $(0, 1)$, the space is shrunk, for negative values of $k$ the space is mirrored around the origin and scaled by a factor of $k$. 

* **Scaling along a single dimension** can be done with a matrix where all elements are same as the identity matrix, but a single element on the diagonal has a scaler value of $k$. This matrix will preserve sizer on all dimension except for the modified dimension, where a scale by $k$ will be applied. Setting a diagonal element to $-1$ will mirror around that dimension.

* **Shear operation** is a matrix which is same as an identity matrix, except a single off-diagonal element, which is a non zero $k$ real number. For example in two dimensions a sheer matrix $S = \begin{pmatrix}
  1 & k \\
  0 & 1 
 \end{pmatrix}$. Shear matrices have a determinant of $1$

* **Orthogonal matrix** is a matrix whose vectors fulfill two conditions:
    * each vector in the matrix (column) has a norm of 1
    * all vectors in the matrix are orthogonal to each other (dot product between any two columns are $0$)

   This means each vector has size $1$ and perpendicular to all the other vectors. The identity matrix $I_{n,n}$ and any other matrix which is a rotation or mirroring in $n$ dimensions of the identity matrix are orthogonal. Applying this matrix as a transformation will result in rotation or mirroring or a combination of both.

* **Proper orthogonal matrix** adds one more condition to orthogonal matrix, that the angles of vectors are preserved. This can be verified by a determinant of $1$. Proper orthogonal matrix will apply a rotation around the origin to the vector space without mirroring. As we have mentioned, mirroring even number of times results in a transformation which is same as a rotation.

These transformation can be applied to images as well, where we can calculate the position of each pixel in a resulting image and apply interpolation or smoothing to fill in the gaps. An exception might be rotation, which is mostly done using a system called **quaternions**. Rotation with matrices suffer from a limitation called gimbal locks, where angles can align and lose degrees of freedom. Combined with limited precision of float representation of numbers, results in highly unstable motion. Quaternions use a four dimensional unit hyper-sphere to describe rotation in three dimensions.

## **Nonsquare matrices**

A square matrix with $n$ rows and $n$ columns applies a transformation in an $n$ dimensional vector space. A matrix with $n$ columns and $m$ rows makes a transformation from an $n$ to an $m$ dimensional space. The input is an $n$ dimensional vector and the output is an $m$ dimensional vector.

When $m < n$, it's called a projection. When $m > n$ the result is higher dimension, but because matrices describe linear transformation, the result of the transformation remains an $n$ dimensional hyper plane in the $m$ dimensional space.

Matrix transformation is done similarly to square matrices.

$$V_m = M_{m,n} \cdot V_n \\ = \begin{pmatrix}
  x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\
  x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  x_{m,1} & x_{m,2} & \cdots & x_{m,n} 
 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} \\ = \begin{pmatrix}
  x_{1,1} v_1 + x_{1,2} v_2 + \cdots + x_{1,n} v_n  \\
  x_{2,1} v_1 + x_{2,2} v_2 + \cdots + x_{2,n} v_n  \\
  \vdots \\
  x_{m,1} v_1 + x_{m,2} v_2 + \cdots + x_{m,n} v_n  \\ 
 \end{pmatrix}$$

## **Matrix multiplication**

Matrices, which describe linear transformations in vector spaces, can be combined. We can express as a single matrix the transformation described by the matrix $B$ happening after a transformation described by the $A$. This is called **composition** of two transformations and the mathematical operation for composition is **matrix multiplication**: $B \cdot A$. When we apply a matrix as a transformation to a vector $AV$ we put the matrix on the right hand side, we can imagine that composition is $B(AV) = BAV$. We can remove the parenthesis because matrix multiplication is associative: the same transformations are being applied in the same order even if we evaluate the multiplications in different orders. This notation of inversed order come from function notations $g(f(x))$

The formula for matrix multiplication uses the formula of matrix vector multiplication (Formula II.2). Each column $k$ in the output matrix, if treated as a vector, is the $k$ th column in the right hand side matrix, also treated as a vector, transformed  (multiplied) by the left hand matrix. Summarized the product formula looks like this:

$$B \cdot A = \begin{pmatrix}
  b_{1,1} & b_{1,2} & \cdots & b_{1,m} \\
  b_{2,1} & b_{2,2} & \cdots & b_{2,m} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  b_{p,1} & b_{p,2} & \cdots & b_{p,m} 
 \end{pmatrix}\begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n} 
 \end{pmatrix} = \\ = \begin{pmatrix}
  \sum_{i=1}^m b_{1,i}a_{i,1} & \sum_{i=1}^m b_{1,i}a_{i,2} & \cdots & \sum_{i=1}^m b_{1,i}a_{i,n} \\
  \sum_{i=1}^m b_{2,i}a_{i,1} & \sum_{i=1}^m b_{2,i}a_{i,2} & \cdots & \sum_{i=1}^m b_{2,i}a_{i,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  \sum_{i=1}^m b_{p,i}a_{i,1} & \sum_{i=1}^m b_{p,i}a_{i,2} & \cdots & \sum_{i=1}^m b_{p,i}a_{i,n} 
 \end{pmatrix}$$

Matrices can be multiplied only if the number of columns of the left hand matrix $m$ is equal to the number of rows to the right hand matrix: the output dimension of the first transformation has to be the same as the input dimension of the second transformation. The input dimension of the product $B \cdot A$ is the number of columns $n$ of the right hand side matrix $A$, the output dimension is the number of rows $p$ in the left hand matrix $B$.

Matrix multiplication is not commutative. For non square matrices, the input-output dimensions need to match up, but for square matrices depending on the order of transformations applied, the result might be different:

$$\exists A, B \in R^{n,n} \rightarrow B A \neq A B$$

In case of square matrices, the determinant of matrix multiplication is equal to the product of the determinants:

$$\det(B \cdot A) = \det(B) \det(A)$$

While the mathematical proof of this is difficult, the intuition behind this is that the rate of change $\det(B \cdot A)$ on the vector space described by the composite transformation $B \cdot A$ is same as the rate of change $\det(B)$ done by the matrix $B$ on the rate of change $\det(A)$ done by $A$.

## **Inverse matrices**

As matrices describe transformations, the inverse of a transformation can be described by the **inverse matrix**. We note inverse matrix of $M$ with $M^{-1}$:

$$M^{-1}M = MM^{-1} = I$$

Applying both $M$ and the inverse $M^{-1}$ is equivalent to applying the identity matrix as a transformation.

Projections don't have a defined inverse (there is loss of information, e.g the matrix might project every line to a point, so the inverse would be expanding a point to a line, which cannot be done with a linear operation, so non square matrices don't have definition of inverse matrices. Similarly we have seen that determinant of $0$ also constitutes as projection, so matrices with $\det(M) = 0$ also do not have an inverse matrix defined.

Several highly optimized algorithms have been proposed to calculate the inverse. In many cases, the inverse matrix is not even calculated, rather the operation done with the inverse matrix (e.g matrix vector multiplication) is calculated or approximated trough an iterative process. This approach saves computation as well as memory for large matrices.

## **Change of basis**

In the same way we applied a matrix as a transformation to a vector in a vector space, we can apply a matrix as a transformation to the base of the vector space as well. Transforming the vector to another basis is done the same way as applying transformation to the vector.

$$U = B \cdot V$$

where $V$ is the vector, $B$ is the transformation matrix and $U$ is the vector $V$ under the basis $B$ (it's the same equation as transforming $V$ by $B$, but different interpretation). 

In some cases transforming a vector under a specific base is more simple than in another base. A common process in linear algebra is to transform a vector to another base (noted with $B$), apply a specific transformation under the base of $B$ (noted with $M_B$), and reverse the base transformation. A transformation can be reversed using the inverse matrix $B^{-1}$:

$$U = B^{-1} \cdot M_B \cdot B \cdot V$$

In the above there are the following steps, which we can read right to left:
* Transform $V$ to another basis with $B$
* Apply matrix transformation using $M_B$
* Reverse basis transformation with $B^{-1}$

All the above steps are simple matrix multiplications.

## **Eigenvectors and eigenvalues**

The line a non-zero vector rests on is called the **span** of the vector. When we apply a matrix transformation to all vectors of a vector space, most vectors would change direction where they point, we say they change their span.

**Eigenvectors** of a matrix $M$ are vectors $V$ of the vector space that do not change their span during vector transformation. The rate of change $\lambda$ of the eigenvectors during the matrix transformation is called **eigenvalue** of the eigenvector. The mathematical relationship can be expressed as:

$$MV = \lambda V$$

All solutions of the above equation for $V$ are the eigenvectors and solutions of $\lambda$ are the eigenvalues. We can rearrange the equation by introducing the identity matrix on the right side and moving everything to the left:

$$MV - I \lambda V = 0 \\
(M- I\lambda) V = 0$$

Writing the above in detail:

$$\begin{pmatrix}
  m_{1,1} - \lambda & m_{1,2} & \cdots & m_{1,n} \\
  m_{2,1} & m_{2,2} - \lambda & \cdots & m_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  m_{n,1} & m_{n,2} & \cdots & m_{n,n} - \lambda 
 \end{pmatrix} \begin{pmatrix} v_{1} \\  v_{2} \\  \vdots\\ v_{n} \end{pmatrix}
= \begin{pmatrix} 0 \\ 0 \\  \vdots\\ 0 \end{pmatrix} $$



* When we apply a scaling matrix, all vectors of a vector space will be eigenvectors

## **Spectral decomposition**

## **Singular value decomposition**
