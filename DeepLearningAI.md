### Activation Function in Neural Network

## **[ReLu] (Rectified Linear Unit)**

$$
a = max(0,z)
$$



f(x) = x (x > ), 0 (x <= 0)

ß0보다 작으면 0, 크면 그대로.
$$
f(x) = x, \ if \ x > 0 \\
f(x) = 0, \ otherwise
$$

### **ReLu를 사용하는 이유?**

다른 activation function들 보다 학습 속도가 더 빠르다. 

기울기가 0과 차이가 많이 나기때문.

**1. Sparsity (벡터를 표시하는 값들 중 0이 많은 것) - 활성화 함수값이 0보다 작을때 (one hot vector)**

반대: dense (벡터를 표시하는 값들 중 0이 별로 없는 것) (embedding vector)

**왜 Sparsity가 유용한지?**

활성화 값이 0보다 작은 뉴런들이 많을수록 더욱 더 sparse한 모습을 띄게 된다. 하지만 기본적으로 neural network에서 사용해온 **sigmoid 함수는 항상 0이아닌 어떠한 값(eg. 0.2, 0.001, 0.003 ...)을 만들어내는 경향이 있어** dense한 모습을 보이게 된다.

**뉴런의 활성화값이 0인 경우, 어차피 다음 layer로 연결되는 weight를 곱하더라도 결과값은 0이 되어 계산할 필요가 없기때문**에 sparse한 형태가 dense한 형태보다 더 연산량을 월등히 줄여준다. (하지만 0으로 한번 할당되면 다시 활성화되지 않으므로 해당 뉴런을 dead neuron / dying ReLu라고 표현하기도 한다.)

**2. Vanishing Gradient - 활성화 함수값이 0보다 클때**

Sigmoid를 사용할 경우, backpropagation은 chain rule을 사용하기 때문에, sigmoid함수의 특성상 값이 0~1이되기때문에 결국에 값들이 layer를 통해 sigmoid함수를 지나갈수록 작아져 0에 converge(수렴)하게 된다.

**결국 layer깊이가 깊어질수록 값이 굉장히 작아져서 결과에 영향을 끼치지 않아** 예측하기 힘든 현상을 가져오게됨.

Sigmoid함수는 좌우로 나갈수록 기울기 (gradient)가 0에 converge하게된다. 그러므로 learning speed가 매우 느리다. layer가많을수록 심각해짐.

### ReLu의 단점?

z가 음수일 때 미분값이 0인 것.

## Supervised Learning with Neural Networks

Types of Supervised Learning with NN

Standard Neural Network: Real Estate, Online Advertising

CNN (Convolutional Neural Networks): image

RNN (Recurrent Neural Network): Sequence data (audio) - time series, Language (translation)

Structured Data: Database

Unstructured Data: audio, image, text

## Logistic Regression - Binary Classification

Linear Regression $$\to\hat{y}= w^tx+b$$

Logistic Regression = $$Sigmoid(z) \ where \ z=(w^tx+b)$$

$$Sigmoid = 1/(1+e^{-z})$$

If z is large positive, $$Sigmoid(z) = \ close \ to \ 1$$

If z is large negative, $$Sigmoid(z) = \ close \ to \ 0$$

Logistic Regression에서 Loss Function을 Squared Error를 사용해도되지만, 나중에 Gradient Discent를 사용하면 Local Minimum밖에 찾지 못한다.

그러므로 Logistic Regression에서는 다음과 같은 Loss Function을 사용한다.

Loss Function -> 각각의 Training Example에 관한 Loss

$$L(\hat{y},y) = -(ylog\hat{y}+(1-y)log(1-\hat{y}))$$

$$if\ y=1: L(\hat{y},y)=-log\hat{y}$$ <- 최대한 크게

$$if\ y=0: L(\hat{y},y)=-log(1-\hat{y})$$ <- 최대한 작게

Cost Function -> 전체 Training set에 관한 Loss

$$J(w,b) = 1/m \sum{L(\hat{y}^{(i)},y^{(i)})}=-1/m\sum{(y^{(i)}log\hat{y}^{(i)}+(1-y^{(i)})log(1-\hat{y}^{(i)}))}$$

### Gradient Descent

We want to minimize the above J (cost function = average of sum of loss function).

find w, b that minimize J(w,b)

W를 계속 업데이트하면서 0으로 converge하도록 만들어야함.

Updating W = W - a(dw), dw = derivative of w

d*J(w,b) / d*w의 의미는 w방향으로 J(w,b)함수가 얼마나 기울었는지 나타낸다. 

W를 미분한 dw는 현재 w의 기울기를 나타내는데 이 기울기가 양수라면 W를 기울기 만큼 빼줌으로써 가장 낮은 구간으로 내려가게됨.

a는 learning rate이다. learning rate는 한 point에서 미분하여 구한 기울기가 너무 크면 W를 업데이트할때 너무 큰 변동성을 주기때문에 변동성을 줄이기 위함이다.

Updating b도 똑같다.

### Derivative (Slope) 기울기

​	Slope = 기울기 = Derivative = 미분

= height/width

### Backpropagation (역전파)

Chain Rule을 이용하여 미분 값을 얻을 수 있다.

Forward Propagation할때 Local Gradient를 미리 계산해둔다.

저장해둔 Local Gradient와 Back Propagation하면서 구한 Global Tradient를 곱하여 최종 미분 값을 구한다.

### Vectorization

Weight와 bias를 업데이트하는 코드를 작성할때 필요한 for loop을 없애기 위한 기술.

for loop으로 계산하는 것보다 vectorization으로 (np.dot(w,x)) 계산하는 것이 훨씬 빠름

CPU and GPU 둘다 SIMD - Single Instruction Multiple Data 기술을 사용하는데, 이것은 python이 parallelism을 빨리 수행하게 도와준다. 

Vectorize해서 계산하면 paralle하게 계산함으로 빠르다.

### Broadcasting

numpy에서 vector계산할 때 빠르게 해주는 기술

(m,n)차원 벡터에 (1,n)차원 벡터를 더하면 자동으로 (1,n)벡터에 m을 곱해서 (m,n)으로 만들어 더해주는 것.





## Activation Function

탄젠트h (tanh)가 sigmoid보다 언제나 좋은 성능을 보인다.
$$
tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$
tanh는 -1 과 1사이의 값을 같는데 이는 0을 평균값으로 가져서 데이터를 중심으로 위치시키는 효과가있기때문

- 다음 층의 학습이 더 쉽게 이루어짐.

Sigmoid function은 더이상 사용되지않는다. tanh가 언제나 우월함.

하지만 output layer에서는 sigmoid사용하여야함

왠만하면 ReLu를 사용함.



### Pros and Cons of Activation Functions

- sigmoid: dont use it unless output layer
- Tanh: superior to sigmoid
- ReLu: most commonly used
- Leaky ReLu: maybe

딥러닝은 non-linear activation function을 사용함.

선형함수는 y 예측값이 Real number일때 출력층에서 사용 할 쑤도있음.



### Derivatives of Activation Functions

$$
sigmoid(z)=a(1-a)\\
tanh(z)=1-(tanh(z))^2=1-a^2\\
ReLu(z)=max(0,z) = 0,\ if\ z <0, \\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 1,\ if\ z > 0,\ \\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ undefined\ otherwise\\
Leaky\ ReLu(z)=max(0.01z,z) = 0.01z,\ if\ z <0, \\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 1,\ if\ z > 0,\ \\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ undefined\ otherwise
$$

### Initializing W to all Zeros causes problem

- hidden units become completely identical
- 2개의 숨겨진 유닛 모두 똑같은 함수를 산출, 결과값 유닛에 똑같은 영향을 줌.
- Induction으로 증명가능



## Week 3 Quiz

The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. 

You decide to initialize the weights and biases to be zero. Which of the following statements is true?

- Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons. 