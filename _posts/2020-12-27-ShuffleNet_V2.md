---
title:  "[논문 리뷰] ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
excerpt: "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"

categories:
  - Deep Learning
tags:
  - Vision
classes: wide
last_modified_at: 2020-12-27T00:10:41+09:00
---
__[arxiv link](https://arxiv.org/pdf/1807.11164.pdf)__  

ShuffleNet V2에서는 경량화 관련해서 보다 현실적인 면에 집중하기 시작한다. 기존의 많은 모델 경량화 논문들은 부동소수점 연산 갯수 (FLOPs) 을 기준으로 모델의 연산 능력을 평가하였는데, 이는 실제 디바이스에 deploy하고 나서 속도 및 지연시간(latency)와 정확히 매칭되지 않는다.

이론적인 연산량과 실제 구현에서의 차이는 다음의 두가지 이유에서 기인한다.
1. __메모리 접근 비용__ (memory access cost, MAC) 및 병렬화 정도  
   - Group convolution의 경우 GPU 연산시 bottleneck이 됨
1. __동작하는 platform__  
   - Tensor decomposition 오히려 CUDNN library을 이용한 GPU 연산에서 ($3 \times 3$ 연산에 최적화) 오히려 연산 속도의 감소를 가지고 옴. 

저자들은 이러한 부분에 집중하여, 효율적인 네트워크 디자인을 위한 4가지 가이드라인을 제시하고, 이를 기반으로 기존 Shufflenet을 개선한 Shufflenet v2을 소개한다.

## __1. MAC 을 최소화 하기 위해 채널 크기 일치__
저자들은 수학적으로 MAC을 정의하는데, 이는 다음과 같다

$$ MAC = hw(c_1 + c_2) + c_{1}c_{2}$$

여기서 $h$와 $w$의 feature map의 spatial size이고, $c_1$과 $c_2$는 각각 입력, 출력 채널 크기이다. 평균값 정리를 정의하면, 위식은 다음과 같이 표현 가능한데,

$$ MAC \geq 2 \sqrt{hwB} + \frac{B}{hw}$$

여기서 $B = hwc_1 c_2$ 로 $1 \times 1$ convolution의 FLOPs이다.

결국, 이론적으로 생각해보면, $c_1$ 과 $c_2$가 같을 때, MAC는 최소가 되고, 실질적인 속도 향상을 가지고 올 수 있다. 저자들은 GPU  (GTX1080ti) 와 ARM (Qualcomm Snapdragon 810) 에서 입출력 채널 크기의 비율을 변경해가면서 속도 측정을 하였고, 성능은 이론적으로 예상한 바와 두 채널의 크기가 같을 때 가장 성능이 우수했다.
![Results of guideline 1](/assets/images/2020-12-27-ShuffleNet_V2/shufflenet_v2_guideline1.jpg)

## __2. Group convolution 으로 인한 MAC 증가가 발생__
ResNext, shufflenet 에서 주요 성능 향상의 key였다 group convolution의 경우 기존의 dense convolution 보다 FLOPs을 줄이며, 오히려 많은 채널 사용을 가능하게 하여 모델의 성능을 높였었다. 하지만 MAC 관점에서 이는 오히려 역효과를 가지고 온다.

$$ MAC = hw(c_1 + c_2) + \frac{c_1 c_2}{g} = hwc_1 + \frac{Bg}{c_1} + \frac{B}{hw}$$

여기서 $g$ 는 group 갯수를 의미하고, $B=hwc_1 c_2 /g$ 로 FLOPs을 의미한다. 동일한 FLOPs ($B$) 을 유지한다고 생각해 보면, $g$의 값이 증가하면 할 수록, MAC는 증가하게 된다. 결국 Group convolution의 경우 traget platform과 task에 따라 신중하게 선택되어야 한다.
![Results of guideline 2](/assets/images/2020-12-27-ShuffleNet_V2/shufflenet_v2_guideline2.jpg)

## 3. __Network fragmentation은 병렬화를 방해__
GoogleNet 시리즈와 NAS 를 통해 만들어진 네트워크에서는 "multi-path" 구조를 많이 채택하였다. 큰 하나의 operation이 아닌 여러개의 작은 opeartions을 활용하게 되면, 정확도 관점에서 많은 이득을 보여줬다. 하지만 GPU와 같은 강력한 병렬화를 지원하는 상황에서 이러한 "파편화된" 연산은 오히려 효율성을 떨어뜨린다. 실제 kernel 실행과 동기화에 많은 overhead가 발생하기 때문이다.

저자들은 이러한 부분을 실험적으로 검증하기 위해, 1~4 단계로 $1 \times 1$ convolution을 파편화해 보았다. 아래 그림에서 확인할 수 있듯이 파편화된 구조에서 GPU의 경우 성능 열화가 심하게 발생하는 것을 확인 할 수 있다. 반면에 CPU 연산만 주로 사용하는 ARM 에서는 상대적으로 이러한 열화가 심하지 않았다.
![Results of guideline 3](/assets/images/2020-12-27-ShuffleNet_V2/shufflenet_v2_guideline3.jpg)

## 4. __Element-wise 연산은 무시할 수 있는 수준이 아님__
실제 ReLU, Add Tensor, Add Bias와 같은 element-wise operation은 작은 FLOPs을 가지지만, 실제 구현에서는 높은 MAC을 가진다. MobileNet과 ShuffleNet에서 약 5~20% 정도의 runtime 시간을 차지할 정도로 큰데, 특히 GPU연산에서 많은 부분을 차지한다.
실제 ResNet BottleNet 구조에서 이 부분을 실험해 본 결과 ReLU 연산과 shortcut을 제거한 것 만으로 아래표에 나타났듯이 약 20% 의 속도 향상을 이뤄냈다.
![Results of guideline 4](/assets/images/2020-12-27-ShuffleNet_V2/shufflenet_v2_guideline4.jpg)

## __Guideline에 맞게 설계된 shufflenet v2__
![shuffltnet v2 units](/assets/images/2020-12-27-ShuffleNet_V2/shufflenet_v2_block.jpg)

shufflenet v2의 경우 어떻게 shufflnet v1에서 제시된 guideline에 따라 변경되었는지 확인해보자
1. 먼저 시작에 channel split을 적용 (group convolution의 효과를 완전히 무시할 수는 없으니)
1. G3 기준으로 하나의 branch는 특별한 연산 없이 shortcut으로 연결. (파편화된 계산을 막기 위해)
1. G1 기준으로 다른 branch에서는 입력과 출력의 크기를 일치
1. 두 branch 결과의 concatenation 후, shufflenet v1에서 핵심이 었던 channel shuffle을 적용
1. 이후 바로 다음 unit과 합쳐진다, 합쳐진 두 unit을 통해 보면, "Concat", "channel shuffle", "channel split"이 하나의 element-wise 연산으로 합쳐 지고 이는 결국 G4 기준으로 element-wise 연산 전체 갯수를 줄이는 효과를 가지고 옴
1. Spatial downsampling의 케이스는 약간의 변형을 통해 유사하게 만들 수 있다. (위 그림 (d))

Shufflenet v2는 연산 효율뿐만 아니라 정확도 관점에서도 향상을 도모 할 수 있다고 저자들은 주장하는데, 2가지 이유를 제시하고 있다.
1. 효율적인 구조로 인해 보다 큰 feature 채널 갯수를 사용하게 됨
1. Channel split로 절반은 바로 다음 block으로 연결되는 효과가 있음 → 이는 DenseNet이나 CondenseNet에서 말하는 핵심 원리인 feature reuse와 관련

여느 논문들과 마찬가지로 ImageNet 데이터에 대해 다른 모델들과 성능 평가를 진행하였다. 이전 모델인 shufflenet v1, 경쟁자인 MobileNet v2, Xception 그리고 DenseNet과 성능 비교를 진행하였다. Shufflenet이 정확도 면이나 Inference 속도 면에서 위의 다른 모델보다 성능이 우수했다. 한가지 흥미로운 점은, 성능 평가시 MobileNet v2 대비 향상된 부분을 많이 강조하는데, 역시 두 모델간 경쟁심리를 엿볼 수 있었다.

단순히 경량화 관점 뿐만 아니라, SENet과 같은 추가 모듈과의 호환성도 좋음을 보여주었고, 또한 Object detection task에서의 성능도 우수함을 보여줌으로써, classfication에 한정된 것이 아닌, 일반적인 task에서도 좋은 성능을 낼 수 있음을 간접적으로 보여주었다.

경량화와 관계없이 한가지 실험 말미에 언급한 부분이 있는데, Object detection task에서 Xception이 shufflenet v1과 mobilenet v2 보다 좋은 성능을 보여주었는데, 저자들은 그것이 훨씬 더 큰 receptive field을 사용했기 때문이라 생각했다. 실제  shufflenet v2에서도 $3 \times 3$ depthwise convolution을 더 추가해서 receptive field을 키우니 정확도가 더 올라갔다고 한다. object detection task에서 receptive field의 중요성을 단적으로 보여주는 예가 아닌가 싶다.