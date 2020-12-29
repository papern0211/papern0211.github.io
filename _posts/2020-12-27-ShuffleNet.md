---
title:  "[논문 리뷰] ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"
excerpt: "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"

categories:
  - Deep Learning
tags:
  - Vision
classes: wide
last_modified_at: 2020-12-27T00:10:40+09:00
---

__[arxiv link](https://arxiv.org/pdf/1709.01507.pdf)__  

ShuffleNet은 ResNet 기반에 group convolution과 channel shuffle이라는 방법을 도입하여, 계산이 효율적인 CNN 구조를 제안한 방법이다. 

## Group convolution
이미 AlexNet에서 group convolution의 개념을 소개했었는데 당시에는 부족한 GPU 자원을 사용하기 위해 궁여지책으로 사용했다고 한다면, ResNext에서 그 효용성이 잘 드러나게 되었다. ShuffleNet에서도 마찬가지로 이러한 group convolution을 사용하는데, 한가지 특징은 MobileNet에서 사용한 Depthwise seperable convolution과 함께 사용하여 그 효과를 극대화 하였다.

## Channel shuffle
Group convolution은 한가지 문제가 발생하게 되는데, 그룹마다 본인 그룹에 관련된 입력에 대해서만 연관되어 출력을 표현하게 되고, 이는 채널간 정보의 흐름을 오히려 막게 되어, 모델이 제대로된 표현을 방해하게 된다. 이를 위해 
그룹간 채널을 셔플링 하는 작업을 넣게 되는데, 이를 통해 모든 채널에 대해 입력과 출력이 모두 연관되도록 할 수 있다.

아래 그림에 Group convolution과 channel shuffle을 잘 나타내고 있다. 한가지 주목할 점은, channel shuffle의 경우 아주 간단한 연산을 통해 구현이 가능한데, group convolution을 통해 얻어진 $N(=g \times n)$개 채널의 출력을 먼저 $(g, n)$ 형태로 __reshape__ 을 하고, 이후 __transposing__ 및 __flattening__ 연산만 해주면 간단히 channel shuffle을 구현할 수 있다. 이러한 연산은 특히 그룹간 채널수가 달라도 문제가 없을 뿐만 아니라, 미분이 가능하기에 end-to-end 학습에도 적용 가능하다
![shufflenet 동작 방법](/assets/images/2020-12-27-ShuffleNet/shufflenet_channel_shuffle.jpg)

## ShuffleNet Unit
저자들은 기본적인 shufflenet unit을 제안하는데 이는 아래 그림과 같다.
![shufflenet units](/assets/images/2020-12-27-ShuffleNet/shufflenet_units.jpg)
ResNet bottleneck 구조에서 출발하는데, 구조의 변경을 다음과 같이 진행했다.
- $3 \times 3$ depthwise convolution 변경
- $1 \times 1$ group convolution 변경 및 channel shuffle 추가
- $3 \times 3$ depthwise convolution 의 ReLU 제거 ($\because$ Xception에서 설명)

Stride와 같이 사용하는 경우, 즉 feature 면적 크기가 줄어드는 경우에 대해서도 Unit을 제안하는데, 다음과 같이 2가지를 변경하면 된다.
- $3 \times 3$ AVG pooling을 shortcut path에 추가
- element-wise addition을 channel cocatenation으로 변경

이러한 shufflenet unit을 활용해서, 전체적인 shufflenet 구조를 저자들은 제안하는데 아래 표와 같다
![shufflenet architecture](/assets/images/2020-12-27-ShuffleNet/shufflenet_architecture.jpg)
3단계에 걸쳐 shufflenet units을 겹겹히 쌓아서 구성하였고, 각 stage 처음에는 stride=2을 적용하였고, 다음단계로 넘어 갈때 마다 이부분을 감안해 채널 수를 두배로 키워주었다. 

저자들은 또한 group number $g$의 변화를 통해 성능 향상을 보여주려 했는데, group number을 크게 할 수록 결과적으로 동일한 complexity에서 네트워크가 많은 채널을 가지게 된다. 이는 보다 많은 정보를 담을 수 있게 되고, 곧 성능 향상을 가져온다. 결국 __"주어진 complexity안에서 채널 크기를 충분히 확보할 수 있고 이는 결국 성능 향상으로 연결된다"__ 는 점이 바로 shufflenet의 가장 큰 성과이지 않나 생각된다.

저자들은 ImageNet 데이터셋에서 shufflenet이 훨씬 적은 연산량을 사용하고도 높은 성능을 보여줌을 보여준다. 또한 COCO dataset을 이용한 Object detection task에서도 MobileNet 대비 더 좋은 성능을 보여주는 것을 보여줌으로써, 본인들이 제안한 네트워크의 효용성을 잘 어필했다.(자세한 성능은 논문 참고)

마지막으로 저자들은 실제 ARM platform을 가지는 mobile device에 shufflenet 모델을 deploy하여 inference 속도를 체크하였다. $g=3$을 기준으로 테스트 해보았을 때, 메모리 엑세스 및 추가적인 오버헤드로 인해 이론적인 연산량 감소 수치 대비 실제 낮은 속도 향상은 보여주었지만, 유사한 성능 내는 Shufflenet 0.5x 기준으로 AlexNet에 대비 13배 빠른 속도를 보여주었다. 속도 향상 결과도 당연히 놀랍지만, 실제 mobile device에 deploy해서 속도 측정까지 했다는 점에서 저자들의 노력이 돋보이는 대목이다.