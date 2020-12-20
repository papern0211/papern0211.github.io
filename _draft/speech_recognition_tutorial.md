# Speech Recognition based on deep learning
음성인식 (Speech recognition)이란, 사람이 말하는 __음성 언어__ 를 __문자 데이터__ 로 전환하는 처리를 말한다. 쉽게 말해서 __받아 쓰기__ 이다. 처리된 문자 데이터는 이후 자연처 처리 (Natural Language Processing)를 통해 그 의미를 분석하고, 다양한 어플리케이션에 적용 가능하다.

- [Connectionist Temporal Classification (CTC)](#connectionist-temporal-classification-(ctc))

## [Connectionist Temporal Classification (CTC)](https://dl.acm.org/doi/pdf/10.1145/1143844.1143891)
일반적으로 Sequnece labeliing task 에서는 학습을 위해 잘 정제되고, 미리 잘 구분된(segmented) 입력 데이터가 요구된다. 하지만, 이러한 작업은 상당히 많은 시간과 비용을 요구함과 동시에, Segmetation 된 결과에 따른 학습 모델의 성능이 영향을 받게 된다. Speech recognition에서 사실 무수히 많은 데이터가 존재함에도 불구하고, 이러한 Segmentation 과정으로 인해, 제대로 활용을 못하고 있는 실정이다.

본 논문에서는 이러한 Segmented labelling 작업이 생략되고, unsegmented labelling을 활용한 학습 방법을 소개한다. 우선 시작하기 앞서, 용어의 정의가 필요한데, 다음과 같다
- Temporal classification:
  - Unsegmented data sequence에 labelling 하는 작업
- Framewise classification:
  - 각 time-step/frame 마다 독립적으로 labelling을 하는 작업

모델에서는 주어진 입력 unsegmented sequence에 대해, 모든 가능한 label sequence의 확률을 고려하여, 이 중 가장 높은 확률을 가지는 sequence을 채택하는 것이다. 

### From Network Outputs to Labellings
네트워크 출력에서 Labelling 과의 관계를 먼저 정의해보자. CTC 에서는 'blank' 라고 하는 추가 label을 정의하는데, 이는 label 없는 경우 혹은 정확한 label을 정의하기 어려운 경우를 위한 유닛이라고 생각하면 된다.

길이가 $T$인 입력 sequence $\bold{x}$ 가 주어졌을 때, 어떠한 특정한 출력 sequence $\pi$ 에 대한 확률은 다음과 같이 정의 된다.
$$ p(\pi | x) =  $$


## [Listen, Attend and Spell (LAS)](https://arxiv.org/pdf/1508.01211.pdf)

## [Streaming end-to-end Speech Recognition for Mobile Devices](https://arxiv.org/pdf/1811.06621.pdf)

- 실시간 streaming 음성 데이터 처리를 위한 모델 제안
- 제안하는 방법은 RNN-T 기반의 방식
[그림]
- RNN-T와 CTC는 구조적으로 다음과 같은 차이가 있음
  - Prediction network 유무
    - CTC의 경우, 매 frame마다 출력하는 output은 independent 하다고 가정
    - 반면, RNN-T의 경우, 이전의 출력된 history로 부터 현채의 출력을 결정하는데 영향을 줌
  - 이로 인해 RNN-T의 경우는 Joint Network 존재
    - Prediction network와 encoder로 부터 얻은 출력값을 처리하는 Feedforward network 추가
- 모델 구조는 다음과 같다
  - 기본적으로는 RNN-T 구조
  - Encoder에서 각 LSTM 뒤에 Projection layer을 붙임
  - 그리고 LAS 모델에서 사용했던 것과 유사하게, Time-reduction layer을 encoder에 적용
    - 실제 이를 통해 많은 속도 향상을 이뤄냄
    - 또한, CTC와 달리 2nd LSTM 이후에도 reduction을 적용해도 성능 열화가 거의 없었다고 언급
- 학습 효율 및 정확도 향상을 위해,
  - 각 encoder의 LSTM과 prediction network에 layer normalization 적용-> RNN의 hidden state의 dynamics의 안정화
  - 효율적인 Forward-backward 알고리즘을 사용해 TPU에서 학습이 가능했고, 이를 기반으로 속도 향상을 가져와서 보다 큰 batch size로 학습이 가능. 궁극적으로 이부분이 정확도 향상을 가져옴
  (* 그럼 batch size가 클수록 더 정확해지는 것임???)
- Inference 속도 향상을 위해,
  - Prediction network의 계산을 독립하고 caching 기술 도입: 실제 언어모델에서는 동일한 history을 가진 prediction이 생길 확률이 높은데, 이부분을 caching 함으로써, 반복적인 연산을 해결-> 약 50~60% 계산 아낌
  - 또한, Encoder에 time-reduction 전후의 계산을 두개의 thread로 나눠서, 최종적으로는 prediction network와 encoder간의 balance을 이뤄냄-> 약 28% 속도 향상
  (* 실제 on-device 모델 만들 때, 이러한 방법은 매우 유용할 것으로 판단됨)
  - 32bit floating point 파라미터를 8bit fixed point 파라미터로 변환 (tensorflow lite 이용)
  - 8bit 데이터의 multiply-accumulate 연산을 진행할 경우에, 두개의 multiplier의 합은 15bit보다 항상 작아서, 결국 32bit accumulate에 하나 이상의 연산을 수행 가능하다-> 결국 이것을 통해 ARM architecture에서 3배 속도 향상
- 만약, 화자가 발화하는 내용이 어떤 context category에 들어가는지 안다면, 인식 성능을 높일 수 있음: Context biasing
  - 즉석으로 만들어지는 list들을 이용해 FST을 만듬으로써 context biasing을 구현
  - Context biasing을 적용한 경우 WER 성능 향상이 비약적 
    - 앞 부분만 맞고 전체는 다른 케이스들을 해결하기 위해 failure arc (취소선이라고 해야될까) 추가
    - 또한, 잘 정리된 unsupervised train data을 이용해 RNN-T 성능 향상
      (사실 이부분 이해가 쉽지 않은데, 이미 가지고 있는 인식기들로 적합한 단어(context에 관련된)들을 이용해(단, label은 없는), 학습에 사용하고 이를 통해 context-baising의 성능을 높인다는 말인데... )

- 일반적으로 Spoken domain의 결과를 Written domain으로 옮기는 작업은 사용자가 직접 설정한 규칙에 의해서 정해짐 (text normalization)
  - 예를 들어, two twenty one b -? 221b 와 같이 변환이 가능한데, 이러한 부분도 E2E로 학습이 가능
  - 본 연구에서는 NN이나 FST을 이용해 학습
  - 하지만, 문제는 이러한 숫자 데이터-텍스트 쌍을 구하기가 쉽지 않음-> TTS 을 이용해 이러한 데이터를 만듬
  - 모델 학습시, 약 90%의 일반데이터에 이러한 Synthetic 데이터를 10% 추가하여 학습

# 실험 과정 (여기서 부터 읽자)








## [Deep Speech](https://arxiv.org/abs/1412.5567v2)

## [Deep Speech2](https://arxiv.org/pdf/1512.02595v1.pdf)

## [Wav2Vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/pdf/1904.05862v3.pdf)