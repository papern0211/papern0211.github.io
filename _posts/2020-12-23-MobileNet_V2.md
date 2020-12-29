---
title:  "[논문 리뷰] MobileNetV2: Inverted Residuals and Linear Bottlenecks"
excerpt: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

categories:
  - Deep Learning
tags:
  - Vision
classes: wide
last_modified_at: 2020-12-23T00:10:50+09:00
---

__[arxiv link](https://arxiv.org/pdf/1801.04381.pdf)__

2017년에 이어 구글 개발진은, MobileNets v2 을 발표하면서 경량화 관점에서 더 최적화된 구조를 제안하였다. MobileNets v1에서 핵심인 Depthwise seperable convolution은 여전히 사용하는 대신, 구조적인 면에서 새로운 개념을 제시하였다.
- Linear bottlencks
- Inverted residuals

우선 Linear bottlenecks 구조는 Covolution layer 구조 설계시 당연시하게 사용되는 ReLU 연산에 대한 고찰에서 출발하였다.  논무에서는 __manifold of interest__ 개념을 기반으로 설명하는데, 이는 우리가 다루는 layer activations 의 subset이라고 생각하면 된다. 

딥러닝 모델을 통해 이러한 Manifold of interest의 경우는 효과적으로 low-dimensional subspace로 임베딩이 가능하고, 이를 통해 높은 성능을 발휘 할 수 있는데, 이러한 부분에 있어 ReLu 연산을 사용할 경우 두가지 관점에서 고민을 해야 한다.
  1. ReLU 변환 후 manifold of interest가 여전히 non-zero volumn에 있다면, ReLU연산은 linear transform과 일치한다.
  1. 입력 manifold가 입력 공간의 low-dimensional subspace에 놓여야지만, ReLU연산은 온전히 정보를 보전할 수 있다.

참으로 헷갈리는 말이다... 아래 그림을 살펴보면, input manifolds을 충분히 담지 못하는 space에서 ReLU 연산을 적용하면, 정보 손실이 발생하지만, 충분히 큰 space로 relu 함수를 적용하면, 그 손실의 양이 적다는 것을 알 수 있다.
{% capture fig1 %}
![Foo]({{ "/assets/images/2020-12-23-MobileNet_V2/mobilenet_v2_manifold.jpg" | relative_url }})
{% endcapture %}
<figure>
  {{ fig1 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Fig. 1. 저차원 manifold을 고차원으로 임베딩시 ReLU 사용 예제</figcaption>
</figure>

따라서  저차원에서 ReLU을 적용하게 되면 nonlinearity로 인해 많은 정보를 잃게 되기 때문에 (실험적으로 확인), __Linear bottleneck__ layer을 구성해야 된다는 것이다.

그렇다면, ReLU는 이대로 없애버리는 것일까? 아니다. ReLU의 경우 사실상 모델에 nonlinearity을 추가해, 우리가 원하는 결과에 대해 모델이 좀더 잘 묘사할 수 있는 역할을 하는데, 이를 없애자는 것은 꺼려지기도 할 뿐 아니라 ReLU을 사용하는 그간의 모든 연구를 부정하는 것이 될 수 도 있다 (너무 극단적으로 생각하긴 했다...)

어쨌든, 논문에서는 이러한 모델의 nonlinearity을 유지하기 위해 Inverted residual이라는 개념을 가지고 왔는데, 아래 그림과 같다.
{% capture fig2 %}
![Foo]({{ "/assets/images/2020-12-23-MobileNet_V2/inverted_residual_block.jpg" | relative_url }})
{% endcapture %}
<figure>
  {{ fig2 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Fig. 2. Inverted residual block</figcaption>
</figure>

1x1 convolution 이후, expansion factor을 도입해 채널을 확장한 뒤, depthwise seperable convolution을 수행하고, 다시 차원을 줄여준다. 차원이 확장되었을 때는 ReLU activation을 적용하였고, 마지막에 차원이 줄어들었을 때는 activation을 적용하지 않았다. 궁극적으로는 이러한 구조를 통해 연산량은 기존대비 작게 유지하지만, 실제 성능은 오히려 더 향상되는 결과를 가져왔다.

한가지 더 살펴 볼것은 ReLU대신 ReLU(6)을 사용한 것인데, 이는 모바일 디바이스에 적합하도록 Fixed-point precision으로 모델 파라미터를 변경할 때, 보다 Robust한 결과를 얻는데 도움이 된다. 즉, high-precision 연산 모델 대비 low-precision 연산 모델의 성능 열화를 줄일 수 있다.

ImageNet Classification,  Object Detection,  Semantic Segmentation 등 이미지 처리관련해서 다양한 task에 대해 성능 평가를 했는데, 기존 MobileNet v1 대비 향상된 성능을 보여줬을 뿐만 아니라, ShuffleNet, NasNet 등과 비교해서도 우위를 보여줬다. 

특히, Object Detection에서 SSDLite (Single Shot Multibox Detector Lite)와 조합해서 YOLOv2 대비 1/10 수준의 파라미터와 1/20 수준의 연산량을 사용하고도 오히려 높은 성능을 도출하였다는 점에서 인상깊었다.