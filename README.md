# Deep-Learning-Basics

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#binary-classification">Binary-Classification</a></li>
    <li><a href="#logistic-regression">Logistic-Regression</a></li>
    <li><a href="#gradient-descent">Gradient-Descent</a></li>
  </ol>
</details>

# Binary-Classification
In a binary classification problem, the result is a discrete value output.The goal is to train a classifier that the input is represented by a feature vector, 𝑥, and predicts whether the corresponding label 𝑦 is 1 or 0.

#### Example: Cat vs Non-Cat
------------------------------------------------------------
In this case, An image is store in the computer in three separate matrices corresponding to the Red, Green, and Blue color channels of the image. The three matrices have the same size as the image, for example, the resolution of the cat image is 64 pixels X 64 pixels, the three matrices (RGB) are 64 X 64 each.

![image](https://user-images.githubusercontent.com/77977624/171976086-f14f55f2-a5bf-4212-afec-046b640a572b.png)

The value in a cell represents the pixel intensity which will be used to create a feature vector of n-dimension. In pattern recognition and machine learning, a feature vector represents an object, in this case, a cat or no cat.The pixel intensity values will be “unroll” or “reshape” for each color in order to create a feature vector, 𝑥. 

![image](https://user-images.githubusercontent.com/77977624/171976604-fedf236a-b4d9-4537-8416-bdf92b121a24.png)

# Logistic-Regression
Logistic regression is a learning algorithm used in a supervised learning problem when the output 𝑦 are all either zero or one. The goal of logistic regression is to minimize the error between its predictions and training data.
#### Loss(error) Function
-----------------------------------------
The loss function measures the discrepancy between the prediction (𝑦̂(𝑖)) and the desired output (𝑦(𝑖)).In other words, the loss function computes the error for a single training example. Generally, there are two formulas to compute the loss:

𝐿(𝑦̂(𝑖), 𝑦(𝑖)) = 1/2(𝑦̂(𝑖) − 𝑦(𝑖))^2 ----------------------------------(1)

𝐿(𝑦̂(𝑖), 𝑦(𝑖)) = −[ 𝑦(𝑖)log(𝑦̂(𝑖)) + (1 − 𝑦(𝑖))log(1 − 𝑦̂(𝑖))] ----------- (2)

In practice, we are more likely to use the second formula rather than the first one as the first one may have local optimums, which would affact the outcomes of prediction. 
#### Cost Funtion
----------------------------------------------------
The cost function is the average of the loss function of the entire training set. We are going to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function.

![image](https://user-images.githubusercontent.com/77977624/171987201-33ba35eb-bf12-47b5-8810-65708241257f.png)
#### Example:Cat vs Non-Cat
-------------------------------------------
Given an image represented by a feature vector 𝑥, the algorithm will evaluate the probability of a cat being in that image.

                               𝐺𝑖𝑣𝑒𝑛 𝑥 , 𝑦̂ = 𝑃(𝑦 = 1|𝑥), where 0 ≤ 𝑦̂ ≤ 1

The parameters used in Logistic regression are:

• The input features vector: 𝑥 ∈ ℝ𝑛𝑥, where 𝑛𝑥 is the number of features

• The training label: 𝑦 ∈ 0,1

• The weights: 𝑤 ∈ ℝ𝑛𝑥, where 𝑛𝑥 is the number of features

• The threshold: 𝑏 ∈ ℝ

• The output: 𝑦̂ = 𝜎(𝑤𝑇𝑥 + 𝑏)

• Sigmoid function: s = 𝜎(𝑤𝑇𝑥 + 𝑏) = 𝜎(𝑧)= 1/1+ 𝑒^-z

![image](https://user-images.githubusercontent.com/77977624/171987400-48e99873-a058-4b80-803e-b2eb6ff6ea38.png)

(𝑤𝑇𝑥 + 𝑏) is a linear function (𝑎𝑥 + 𝑏), but since we are looking for a probability constraint between [0,1], the sigmoid function is used. The function is bounded between [0,1] as shown in the graph above.

# Gradient Descent
In order to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize cost function J(𝑤, 𝑏) so that implementing logistic regression, we introduce a mathematical strategy called Gradient Descent. The figure below illustrates how Gradient Descent works in finding global optimum of J(𝑤, 𝑏). In the case of Gradient Descent, gradient also as known as derivative which means we are about to take derivative of J respect to 𝑤 and 𝑏.

![image](https://user-images.githubusercontent.com/77977624/174477597-5678616d-11b1-49f3-8eac-8c285133344d.png)

To have a better insight into Gradient Descent, we make mathematical analysis along 𝑤 axis and 𝑏 axis respectively. The next figure represents the correlation of cost function J with 𝑤. From which, it is noted that we can find 𝑤 that corresponding to the optimum of J by updating 𝑤 iteratively. We follow the formula below to update 𝑤: 𝑤 := 𝑤 - α * d(J(𝑤))/d𝑤, where α is learning rate. As we can see, the derivative of J respect to 𝑤 is smaller than zero on the right side of the optimum, while it is greater than zero on the left side of the optimum. Thus, 𝑤 is getting closer and closer to the optimum after every iteration.

![image](https://user-images.githubusercontent.com/77977624/174477991-db237ec3-7150-4091-b0e6-994d19471590.png)


