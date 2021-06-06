# GMRES_and_CG
In this study I have investigate the performance of alternative least squares solvers for Ax = b. The numerical techniques to be compared are: "Generalized Minimum Residual (GMRES)", "Conjugate Gradients (CG)", "The Pseudoinverse"


## Introduction

First, I have created 6 “A” symmetric matrixes as stated in the homework and then I have created 30 realizations of the vector b. I have then implemented a Python function to find the following errors. 

![image](https://user-images.githubusercontent.com/48417171/120927197-b406f800-c6e8-11eb-911c-685fed4d0cfa.png)

![image](https://user-images.githubusercontent.com/48417171/120927200-b5d0bb80-c6e8-11eb-896b-d0d21fcaec6e.png)

I have seen that the E_s error finds the error between the predicted x (predicted by using noisy b vectors) and the noiseless x_0 vector. The second error E_ofinds the error between noisy b and the multiplication of the “A” matrix and the predicted x value. Therefore, I have used this error to investigate when the algorithms (CG, GMRES) converge. In other words, as stopping criteria I have used a max iteration number and determine the best iteration number by inspecting where these algorithms converge. 


## Pseudoinverse Implementation

First, I have implemented pseudoinverse by SVD as stated in the lecture notes as follows. 

![image](https://user-images.githubusercontent.com/48417171/120927224-c6813180-c6e8-11eb-8e82-e6c14dadf69b.png)

However, I have seen that the NumPy library has implemented this more efficiently than my own method. Therefore, for performance issues, I have used NumPy’s implemented pseudoinverse method. Furthermore, this algorithm does not have any hyperparameter, so I have directly proceeded to implement other 2 algorithms. 


## Conjugate Gradients (CG) Implementation 

For implementing this algorithm I have used the pseudocode from the lecture notes. The pseudocode is as follows, 

![image](https://user-images.githubusercontent.com/48417171/120927386-6048de80-c6e9-11eb-9f0c-919b83f786dc.png)

This method has been implemented as a python function and then run for the “N” value of 30. In other words for 30 iterations. I have found the E_serror for different noise levels and different “A” matrixes, the result is as follows.

![asdasddsadas](https://user-images.githubusercontent.com/48417171/120927516-bc136780-c6e9-11eb-9595-aece3b83b25c.png)

From the graphics, I can see that as expected the noise levels do not have any effect on the converges of the E_o error. It is because E_o error is the error between noisy b and the multiplication of the “A” matrix and the predicted x value. From these graphs I have determined the number of iterations for different “A” s. So, my stopping criteria is inspecting where the algorithm converges. For example, for “A (100x100) tao:0.01” I will use 9 iterations because after 9 iterations the error does not change much. There are two curves that we cannot see the iteration value they converge. For “A (500x500) tau:0.1” the decrease in the error is linear so I am guessing that in 55 iterations this algorithm will converge but for the “A (10000x10000) tau:0.1” it seems our algorithm is not converging. I will discuss this part later in this report. 

Furthermore, from these graphics, we can see that if we use a smaller number of iterations, we will end up with a vector that is not that close to the “x” vector that we want to predict. So, increasing the iteration number makes our prediction better. Interesting thing is that we do not have to wait for algorithm to converge if we do not want great precision on the predicted “x”. Also, in real life, we know that there will be noise in our “b” vectors. So, after a point of iterations, the change in the predicted “x” vector is not that important because the measured “b” vector is already noisy. Therefore the E_s error converges much earlier than the E_o error to a high value of error. Thus, using a significantly high number of iterations is a waste of time in a perfect environment because the precision we want might be not that high, we can get close to the original “x” in a small number of iterations. In real life, it is not possible to find the “b” vector noiseless so we cannot even get very precise in finding the noiseless “x” vector. Therefore, using a significantly high number of iterations in real life does not make our predicted “x” vector better.   

Also, we can see that the noise effects E_s error. When the noise increases the error increases because we are trying to get close to the noiseless “x” by using the noisy “b” vector. When the noise’s std is bigger than the std of the noiseless “b” vector then it is possible that we will end up with a bad prediction of the noiseless “x” vector. We can see this when we have added 1 std Gaussian noise to the vector “b” where iterations do not decrease the error. From the figures, we can see that the error E_s cannot get lower than the std of the noise. 

Finally, to show the orthogonality I have saved the normalized residuals and in the end, I merged them in the matrix form Q in the CG algorithm. Then I have proceeded with: Q^T Q-I. As a result of this, I am expecting to have a zero matrix. I have tested this approach with 3 different matrixes and the results are as follows.

![image](https://user-images.githubusercontent.com/48417171/120927548-e2390780-c6e9-11eb-923b-bc0cc0fa7d6d.png)

The max, min, and mean values are extremely low. We can take them as zeros and therefore I have proved that the normalized residuals are orthonormal to each other.


## Generalized Minimum Residual (GMRES) Implementation

For implementing this algorithm I have used the following pseudocode. 

![image](https://user-images.githubusercontent.com/48417171/120927603-17ddf080-c6ea-11eb-93ae-c866447a6503.png)

I have used Lancsoz because our matrixes “A”s are symmetric. The process of the Lancsoz is given as following.

![image](https://user-images.githubusercontent.com/48417171/120927617-25937600-c6ea-11eb-9fec-b4801f9dcb17.png)

By Lancsoz I have found orthogonal vectors and replace H_kwith T_k. I have implemented this algorithm in the python function. The only missing thing was the solution to the equation: 

![image](https://user-images.githubusercontent.com/48417171/120927637-3643ec00-c6ea-11eb-829e-8f74f31633f3.png)

I have solved this by decomposing T_(k+1,k) to QR. For decomposing I have inspected that the fastest way is using NumPy’s QR decomposition algorithm. Therefore, I have used it. After the decomposition y_kis found as follows: 

![image](https://user-images.githubusercontent.com/48417171/120927653-44920800-c6ea-11eb-9904-b1d4f870d3bb.png)

In that way, I have estimated the “x” values for different Krylov subspace dimension sizes. This whole process has been implemented as a python function and then run for the “n” value of 30. In other words until k = 30 iterations. I have found the E_s and E_o errors for different noise levels and different “A” matrixes the result is following.

![image](https://user-images.githubusercontent.com/48417171/120927670-570c4180-c6ea-11eb-9168-8dc3000c6cb0.png)

E_s& E_o errors are as expected similar to the CG algorithm. The comments I have made for both errors are also valid for these graphs. By looking at the E_o graphs I have determined at which points I should stop the algorithm. 

Furthermore, from these graphics, we can see again that if we use a smaller number of iterations, we will end up with a vector that is not that close to the “x” vector that we want to predict. So, increasing the dimension K number (i.e., iteration) makes our prediction better. However, after a point, our result does not change much so it is not a good idea to significantly increase the iteration, same as the CG algorithm case. Also, similar to the CG algorithm when the noise’s std is so big we are not able to get a good prediction of the noiseless “x” vector. From the figures, we can see that the error E_s cannot get lower than the std of the noise.

Furthermore, to show that the Krylov subspaces are orthogonal I have returned the Q matrix at the last iteration of the GMRES. Then I have proceeded with: Q^T Q-I. As a result of this, I am expecting to have a zero matrix. I have tested this approach with 3 different matrixes and the results are as follows.

![image](https://user-images.githubusercontent.com/48417171/120927685-64c1c700-c6ea-11eb-8bc7-4f6eb57999bd.png)

Since the max, min, and mean is so low we can conclude that these matrixes are actually zero matrixes and therefore Q is orthonormal.


## Noise vs Es Error

![image](https://user-images.githubusercontent.com/48417171/120927720-89b63a00-c6ea-11eb-8569-72d974a93170.png)
![image](https://user-images.githubusercontent.com/48417171/120927722-8b7ffd80-c6ea-11eb-91b9-5a0e2022761c.png)
![image](https://user-images.githubusercontent.com/48417171/120927726-8e7aee00-c6ea-11eb-963e-3360c76109e4.png)
![image](https://user-images.githubusercontent.com/48417171/120927729-9044b180-c6ea-11eb-95ef-e4d3d70d4009.png)
![image](https://user-images.githubusercontent.com/48417171/120927731-920e7500-c6ea-11eb-8d24-9d75186b2848.png)
![image](https://user-images.githubusercontent.com/48417171/120927732-9470cf00-c6ea-11eb-848a-9462d7eaea2d.png)
![image](https://user-images.githubusercontent.com/48417171/120927734-963a9280-c6ea-11eb-9b04-e5db3b634ab0.png)
![image](https://user-images.githubusercontent.com/48417171/120927737-98045600-c6ea-11eb-86b0-77622ef6cc40.png)
![image](https://user-images.githubusercontent.com/48417171/120927738-99358300-c6ea-11eb-8d94-e6e923f2f4a0.png)
![image](https://user-images.githubusercontent.com/48417171/120927739-9aff4680-c6ea-11eb-8f01-26902966cf59.png)
![image](https://user-images.githubusercontent.com/48417171/120927741-9cc90a00-c6ea-11eb-8d6b-f2f850ed1dca.png)
![image](https://user-images.githubusercontent.com/48417171/120927744-9e92cd80-c6ea-11eb-9b81-80d7986f4494.png)
![image](https://user-images.githubusercontent.com/48417171/120927747-a05c9100-c6ea-11eb-9422-8528514ede8f.png)
![image](https://user-images.githubusercontent.com/48417171/120927750-a2265480-c6ea-11eb-901a-914155670e6a.png)
![image](https://user-images.githubusercontent.com/48417171/120927753-a3f01800-c6ea-11eb-97d8-625ab1dbbc70.png)
![image](https://user-images.githubusercontent.com/48417171/120927757-a5214500-c6ea-11eb-9937-65724f1cc9a8.png)

To get these graphs I have run the algorithms (Pseudoinverse, CG, GMRES) with different ‘A’ matrixes. As a stopping criterion I have first examined Figure 3 and Figure 7 and by inspection determined the iteration number for every algorithm and ‘A’ matrix combination. The algorithm stopped after that iteration has been reached. I have plotted E_s errors with respect to 3 different noise levels and I have also connected those points. Furthermore, both the x and the y axis are in logarithmic form. The corresponding values for the 3 points are also displayed on the graphs. 

Firstly, as the noise increases the error increases. This is because we try to predict the ‘x’ vector from the equation, Ax=b where the noise is applied to these ‘b’ vectors so we get the solution from the noised ‘b’ vector, not the original noiseless ‘b_0’ vector. As the noise increases the difference between ‘b’ and ‘b_0’ vector increases and because this difference increases also the difference between ‘x’ and ‘x_0’ increases. Thus the error increases as the noise increases. 

Furthermore, the log plot of the error and the noise std is linear. That means the std of the noise and the error are linearly dependent. So in real life, as the noise in our measurement of the noiseless ‘b’ vector increases, we will get further away from the noiseless ‘x’ vector. This is not true for big-sized with big tau valued matrixes for CG and GMRES. This is because these matrixes are ill-conditioned so the algorithm is not able to get a good approximation of the ‘x_0’ vector and therefore relation is not linear. I will further discuss this ill-condition in the next section. This is not true for pseudoinverse because pseudoinverse is not iterative. As an analogy, CG and GMRES algorithms try to optimize Ill-matrixes like ML algorithms trying to optimize models with high learning-rates. Like those ML algorithms may diverge the CG and GMRES also may diverge for ill-conditioned matrixes. Further, pseudoinverse has no iteration, it is just a one-step. Therefore we see a linear relation. However, this does not mean that the error we get is small. We are ending with a  very high error with the pseudoinverse when the matrix is ill-conditioned. This is because I think, for the ill-conditioned matrixes, a small change in matrix results in a big change in ‘x’ for the equation, Ax=b. Think it as following, because pseudoinverse tries to find A^(-1) with a good approximation, there are still small differences between A^(-1) and A^+matrixes. These small differences result in a big change in the ’x’ vector. So although the two matrixes A^(-1) and A^+ are close, the ‘x’ vectors (noised and noiseless) are not close, therefore the error results in a big value. We can not take the inverse of a nonsquare matrix but this is a good way of thinking why pseudoinverse is having a bad result. 


## Iteration vs Log(Eo Error) with Different ‘A’s

![image](https://user-images.githubusercontent.com/48417171/120927774-bc603280-c6ea-11eb-95e7-920fd80a667c.png)

![image](https://user-images.githubusercontent.com/48417171/120927780-c08c5000-c6ea-11eb-9fb7-5bb3218ec279.png)

![image](https://user-images.githubusercontent.com/48417171/120927783-c2561380-c6ea-11eb-89a4-8a48d2883743.png)

I have plotted the E_o error with respect to iteration number for different A matrixes. Since the  E_o error is not related to the applied gaussian noise all the three graphics are similar. The only difference comes from the randomly initialized ‘b’ vectors. What we have seen here is that. First of all, it takes more iterations for bigger-sized matrices to converge, simply because of the increased number of rows and columns. Also, at the converges the E_o error is bigger for big-sized matrixes. This is mainly because it is harder to have a good approximation of big vectors. 

Secondly, as the Tau value increases the converges takes more time and if it converges the E_o error is bigger than the same sized but smaller Tau valued matrixs’ error. That is because, as it is stated in the homework, as the Tau value increases condition number of the matrix increases. For the equation Ax=b the high condition number of matrix ‘A’ means ill-conditioned ‘A’, for ill matrixes a small change in ‘A’ results in a huge change in ‘x’. This is why the matrixes created with bigger Tau have a difficulty in terms of converging. I can explain this by analogy with an ML algorithm that its learning rate is defined problematically big. If Tau and size of the matrix are big then the matrix is ill-conditioned, by analogy, we can say that the learning rate is given so big (remember a small change in A results with a huge change in ‘x’) that the algorithm starts to diverge, as we can see in the matrix ‘A(10000x10000) tau:0.1’. Also, big Tau values decrease the sparsity of the matrix ‘A’ as well so it is harder to do the linear algebraic processes.
