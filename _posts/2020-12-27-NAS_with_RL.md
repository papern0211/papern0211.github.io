---
title:  "[논문 리뷰] Neural Architecture Search with Reinforcement Learning"
excerpt: "Neural Architecture Search with Reinforcement Learning"

categories:
  - Deep Learning
tags:
  - NAS
classes: wide
last_modified_at: 2020-12-21T00:21:00+09:00
---

__[arxiv link](https://arxiv.org/pdf/1611.01578.pdf)__

Neural network 모델의 description을 표현하는 RNN을 강화학습을 이용해 학습하여, 최적의 모델을 생성해내는 방법을 제안한 논문으로, 이를 Neural architecture search라고 일컷는다.

이전 SOTA CIFAR-10 대비하여, 0.09% 정확도 향상/1.05배 속도 향상을 이뤄냈으며, 또한 Penn Treebank 데이터 넷에 대해, LSTM을 능가하는 RNN cell을 제안함으로써, SOTA 모델 대비 3.6 peplexity가 향상된 모델을 제안할 수 있었다.

## GENERATE MODEL DESCRIPTIONS WITH A CONTROLLER RECURRENT NEURAL NETWORK
간단한 CNN 을 생성하는 모델을 예로 들어 방법을 설명하는데, 아래 그림과 같이 CNN의 architectural hyperparameters을 생성하는 RNN contoller을 먼저 생각해보자.

![RNN controller가 sample CNN 생성하는 방법](/assets/images/2020-12-27-NAS_with_RL/NAS_RL_CNN_example.jpg)

우리의 목적은 이렇게 해서 생성되는 샘플 CNN의 validataion accuract $R$을 최소화 하는 string을 추출할 수 있도록 RNN의 paramete $\theta_{c}$을 업데이트 하는 것이다. 이를 위해 강화학습을 이용하는데, 우리는 expected Reward $J(\theta_{c})$을 다음과 같이 정의한다:
$$ 
J(\theta_{c})=E_{P(a_{1:T};\theta_{c})}[R]
$$
여기서 $R$은 미분이 불가능한 값이기에, policy gradient 방법을 적용한다. 
$$ 
\bigtriangledown_{\theta_{c}}J(\theta_{c})=\sum_{t=1}^{T}E_{P(a_{1:T};\theta_{c})}[\bigtriangledown_{\theta_{c}}\log P(a_{t}|a_{(t-1;1)};\theta_{c})R] 
$$
실제 계산을 위해 approximation 하면,
$$ 
\bigtriangledown_{\theta_{c}}J(\theta_{c})\simeq\frac{1}{m}\sum_{k=1}^{m}\sum_{t=1}^{T}\bigtriangledown_{\theta_{c}}\log P(a_{t}|a_{(t-1;1)};\theta_{c})R_{k} 
$$
여기서 $m$은 sample의  architecture 갯수이고, $T$는 hyper-parameter 갯수이다.

위의 수식은 unbiased estimate 이지만, 여전히 높은 variance 값을 가진다. 이를 해결하기 위해, 간단히 baseline function을 도입하는데, 다음과 같이 식을 변형 할 수 있다.
$$ 
\bigtriangledown_{\theta_{c}}J(\theta_{c})\simeq\frac{1}{m}\sum_{k=1}^{m}\sum_{t=1}^{T}\bigtriangledown_{\theta_{c}}\log P(a_{t}|a_{(t-1;1)};\theta_{c})(R_{k} - b) 
$$
$b$는 이전 arcituecture의 정확도들의 exponential moving average로 구한다.

최신 CNN에 대해 구조에 대해 적용을 하기 위해서는 한가지 더 고민해야 할 부분이 있다. 바로 ResNet에서 도입된 Skip connection 이다. 이를 위해 attention mechanism을 도입하였는데, $N$ layer 기준으로 $N-1$ 번까지의 layer에 대해 각각 selection 확률을 다음과 같이 계산한다
$$ 
P_{ji} = sigmoid(v^{T} tanh(W_{prev}*h_{j} + W_curr * h_{i}))
$$
여기서 $P_{ji}$는 layer j가 layer i의 입력일 확률이다.

결과적으로 이러한 확률을 추가하여, 마찬가지로 RL을 적용하여 학습할 수 있다. 하지만 이렇게 할 경우 Search Space가 넓어지는 것과 동시에, layer간 연결을 하는데 있어 문제가 발생하다. 즉, 입출력 연결관계가 없는 layer도 출현가능하고, skip connection으로 입력이 연결될 때 사이즈가 맞지 않는 경우도 존재한다. 이를 해결하기 위해 논문에서는 다음과 같은 세가지 장치를 추가했다.
1. 입력이 연결되지 않은 layer는 이미지를 입력으로 연결
1. 출력이 연결되지 않은 layer는 final layer에 연결
1. 입력을 concatenate할 때 사이즈가 다르면 zero-padding을 통해 맞춰줌

조금 더 일반화 하게 되면, 여기에 학습 learning rate도 hyper-parameter로 추가 가능하며, 또한 pooling, batchnorm 등과 같은 구조도 추가 가능하다.

## GENERATE RECURRENT CELL ARCHITECTURES
이제 같은 방식으로 최적화된 RNN cell을 제안할 수도 있다. RNN과 LSTM cell 연산은 tree 구조로 표현할 수 있는데, 각 트리의 노드는 두개의 입력 ($x_t$, $h_{t-1})와 출력 (h_t)로 표현 가능하다. 이를 combination 연산(addtion, elementwise multiplication 등)을 엮어 주면 여러 노드를 연결한 트리를 구성할 수 있다. 또한 LSTM과 같이 memory state ($c_t$)도 추가하고, 이를 앞서 계산한 hyperparameter와 연결을 하도록 하여 구성할 수 있다. 

![RNN controller가 sample RNN 생성하는 방법](/assets/images/2020-12-27-NAS_with_RL/NAS_RL_RNN_example.jpg)

위 그림은 이러한 방법의 예제인데, 간단히 과정을 설명하면 다음과 같다.
1. Tree node index 0: $a_{0}=tanh(W_{1}*x_{t-1} + W_{2}*h_{t-1})$
1. Tree node index 1: $a_{1}=ReLU((W_{3}*x_{t}) \odot (W_{4}*h_{t-1}))$
1. Cell index 0을 두번째로 출력: $a_{0}^{new}=ReLU(a_{0} + c_{t-1})$
1. Tree node index 2: $a_{2}=sigmoid(a_{0}^{new} \odot a_{1})$
1. Cell index 1을 첫번째로 출력: tree 1번의 activation 전의 출력을 사용, $c_t=(W_3 * x_t) \odot (W_4 * h_{t-1}$)

결국 미리 어느정도의 RNN cell에 대한 구조를 세우는 방법(elements)을 정의하고 이것의 combination을 다양하게 학습 하게 하여, 최적의 RNN cell을 찾을 수 있다.

CIFAR-10 데이터에 대해서 제안된 방식으로 생성된 CNN Network 성능을 SOTA 알고리즘과 비교하였고, Penn Treebank dataset에 대해 RNN cell unit 생성 성능을 비교하였다. 실제 최신 SOTA 성능을 보여주는 network에 준하는 성능을 보여주었고, 이 방식의 가능성을 확인 할 수 있었다. (자세한 성능은 논문 참조)

제안된 방법은 많은 딥러닝 개발자들로 하여금 더이상 모델링이 필요없다고 생각하게 할수도 있지만, 여전히 사람이 생각한 기본 module에 대해 그것을 최적화하는 방식의 문제라고 생각된다. 결국 하나의 모듈을 구현하고 기본 연산 단위를 정의하는 것은 여전히 사람이 하는 것이니까 말이다. NAS는 결국 이러한 모델링 과정에서 insight을 제공할 수 있는 좋은 도구이자 동반자(?)로 발전하지 않을까 생각된다.

