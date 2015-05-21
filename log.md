# Finding interestingly related dimensions

1. Start with high dimensional data
2. Determine for each pair of dimensions if they are interestingly related
3. Return the most interestingly related dimensions

### Approach

The general approach will be to train a neural network to do the classification.  The network will be trained on generated labelled data.
Classification with neural net might enable great speed and flexibility with respect to interestingness metric.

### What "interestingly related" means?

It is assumed that dimensions are interestingly related when the $R^2$ with respect to a model is close to 1. It can be also assumed that a very complex model makes the dimensions less interestingly related.
The classification algorithm does not have access to the model.

### Data generation

To obtain the desired neural network, a training dataset has to be generated. Rather than using multidimensional data generation and then dividing it into pairs, multiple 2D samples are generated. $N_d$ samples are generated, each consisting of point coordinates $x_i$, $y_i$ and label $l$. The number of points in each sample is $dim(X)=dim(Y)=N_s$ and the label is just a scalar.
#### Procedure
For each sample the procedure is as follows:

 1. Pick $X$ from uniform distribution: $U(0, 1)$
 2. Pick a model from a simple function generator
 3. Apply function to $X$ to obtain $Y_0$
 4. Pick standard deviation of noise from uniform distribution: $\sigma^2=U(0, \sigma_{max}^2)$
 5. Add Gaussian noise to each point $Y_i=Y_{0,i}+\mathcal{N}(Y_{0,i}, \sigma^2)$ to obtain $Y$
 6. Calculate $R^2$

Then various approaches diverge.
#### Model generation

The models should represent a wide variety of shapes that might be considered interesting. Lines and other continous functions are obvious, but there need to be other non-functional shapes, like circles and squares.
Currently the idea is to generate functions and then to put them in various coordinate systems, so that they are no longer only functions in cartesian system.

It consists of these steps  (very ugly):
1. Generate expression. Use $X$, $Y$ and numbers from normal distribution, joined by operators:  $+, -, \times$.
2. Use the expression as a function with domain $[0, 2\pi]$.
3. Transform coordinates either to cartesian or to polar.

## Regression + fully connected network approach

In this first approach coordinates of points were taken as an input to the network and $1-R^2\in [0, \infty)$ taken as a label. Coordinates were in range $[0, 1]$.

### Network used

Network was created using Lasagne. Two hidden layers of 10 hidden units were used. All units were rectilinear.

### Training

First time the generated data was not properly diagnosed and all labels ended up being almost 0. Nothing useful was learned.

### Conclusion

For now this approach is on hold. In the end convolutional networks seem to be better suited for the task and then discrete domain is required.

## Classification + convolutional network approach

Only $R^2$ will be used as interestingness measure. There will be a threshold, so the label $l\in\{0,1\}$. 0 is not interesting, 1 is interesting.
Value of $R^2$ higher than 0.7 looks like a good threshold for looking interesting.

### Network used

Experimentation was a bit cumbersome, so decided to try out the new tool Nvidia DIGITS. It uses Caffe under the hood and has a nice interface for training deep convolutional neural networks.

The classic LeNet was used for first test. It takes $28\times28$ images and originally has 10 outputs. In this case two outputs were used: true for $R^2>0.7$ and false for smaller values.

### Generating the images

Core generation was the same as with continous domain, but now the points were put into a matrix. The continuous point coordinates had to be discretized.  Each matrix field value is proportional to the number of points assigned to it.
The matrix is than converted into .png image and brightness scaled to maximum range.

#### Why not use discrete domain straight away?
Calculated $R^2$ would be different because of discrete domain. There might be other issues that can prevent generalization to natural datasets.

#### Discretization of samples for convolution

The data has to be discretized into a fixed size matrix for training. It is hard to scale the data in discrete domain due to numeric artefacts. It is best to scale in continuous and do the rounding at the end.

### Training

10000 images were divided into train (80%) and validation (20%) set (DIGITS does not support test set yet). About $\frac{2}{3}$ had the true label and $\frac{1}{3}$ false (should be tuned later). The LeNet has achieved 96.9% accuracy on the validation set. 

![Training progress](https://lh4.googleusercontent.com/vAvWKUSABnllcd42u_gxuxU8yNTcApxrRvYnVc-92IcZH2TwvwhuaXTSnXWFxW8uYu4F1U8TRAIYOr4=w2560-h1538-rw)

It can be seen that training after epoch 12 does not give much improvement.