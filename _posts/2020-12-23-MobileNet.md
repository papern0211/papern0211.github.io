---
title:  "[논문 리뷰] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
excerpt: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"

categories:
  - Deep Learning
tags:
  - CNN
  - Vision
classes: wide
last_modified_at: 2020-12-23T00:10:40+09:00
---

__[arxiv link](https://arxiv.org/pdf/1704.04861.pdf)__

2020년 12월 기준 인용수가 무려 6,000이 넘을 정도로 많은 연구자들이 인용한 논문으로써, 향후 많은 논문들에서 채택한 Depthwise seperable layer을 이용해 경량화를 효율적으로 보여준 연구이다.

앞서 언급했듯이, MobileNets 에서 경량화의 핵심은 바로 Depthwise seperable convolution으로 아래 그림에서와 같이 Batch Normalization과 ReLU을 같이 조합해 구성되었다.
{% capture fig1 %}
![Foo]({{ "/assets/images/2020-12-23-MobileNet/blockdiagram_dsc.jpg" | relative_url }})
{% endcapture %}
<figure>
  {{ fig1 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Fig. 1. 일반적인 convolution과 depthwise seperable convolution blockdiagram</figcaption>
</figure>
일반적인 convolution layer와 연산량 비교를 해보면, 아래 그림에서 알 수 있듯이, 약 kernel 크기의 제곱만큼 연산량의 감소를 이룰 수 있다. 일반적으로 3x3 kernel을 많이 사용하기에, 약 9배정도의 연산량이 감소된다.
{% capture fig2 %}
![Foo]({{ "/assets/images/2020-12-23-MobileNet/flops_mobilenets.jpg" | relative_url }})
{% endcapture %}
<figure>
  {{ fig2 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Fig. 2. 연산량 비교</figcaption>
</figure>
한가지 흥미로운 점은, 논문에서는 단순히 Multi-Adds (FLOPS)로 연산량의 이론적 감소 뿐만 아니라 실제 구현 관련해서도 고민을 했다. 실제 구현에서 연산 속도 향상을 위해서는 general matrix multiply (GEMM) 함수 활용을 해야하는데, 1x1 pointwise convolution은 memory상 재정렬 같은 상황을 고려하지 않고, 바로 GEMM을 이용해 구현이 가능하다는 것이다. 이는 1x1 pointwise convolution이 전체 연산량의 약 95%, 전체 파라미터의 약 75을 차지하며 주된 연산이 되기에, MobileNets의 실제 구현에서 최적화가 매우 잘 이뤄질 수 있음을 간접적으로 보여준다.

이 연구에서는 기존 모델과 비교해 어느정도까지 작은 모델을 만들 수 있고, 실질적으로 그에 따른 정확도와의 정량적 분석을 위해, 두가지 scale factor 개념을 소개했다
- __width multiplier__: convolution layer의 node 갯수 비율 (0~1)
- __resolution multiplier__: 입력 이미지의 축소 비율 (0~1)

위의 scale factor의 변경를 통해 실제 모델의 파라미터/연산량이 어떻게 바뀌고, 그에 따른 정확도는 어떻게 변화되는지를 확인 할 수 있다.

ImageNet 데이터에 대해서 같은 구조에 일반적인 convolution와 Depthwise seperable convolution을 적용한 경우 파라미터와 연산량은 약 8~9배 감소한 것 대비 정확도는 1%정도 밖에 열화 되지 않았다.
{% capture fig3 %}
![Foo]({{ "/assets/images/2020-12-23-MobileNet/accuracy_mobilenets.jpg" | relative_url }})
{% endcapture %}
<figure>
  {{ fig3 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Fig. 3. 연산량 비교</figcaption>
</figure>

또한 Fine Grained Recognition, Large Scale Geolocalizaton, Face Attributes, Object Detection,  Face Embeddings 같은 다양한 task에 대해 MobileNets을 적용한 결과 기존의 baseline 모델 대비 동등한 성능을 보여주었다.
