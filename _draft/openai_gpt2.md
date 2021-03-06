# OpenAI GPT-2
제목: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

2019년 2월에 OpenAI GPT-2가 발표돠었다. 기존의 OpenAI GPT-1과 비교해서 생각해보면 어떤 철학적인 면에서 변화를 이루었다고 생각한다. (개인적 생각이다.) GPT-1에서 pre-training 모델과 이를 기반으로한 supervised fine-tunning의 조합을 통해 다양한 task에서 좋은 성능을 보여주었지만, 여전히 supervised learning을 필요로 한다는 점에서 gpt-2는 고민을 했다고 생각한다.

GPT-2에서는 이러한 관점에서 zero-shot setting 하에서 pre-trained 모델의 파라미터나 구조 변경 없이도 downstream task 수행을 잘 할 수 있음을 보여주고, 특정 과제에 따라서는 SOTA 성능을 보여준다. 

## __Approach__
GPT-2에서 핵심은 바로 Language 모델링이다. 우선, 언어는 그 자체로 sequential한 특징으로 인해, 다음과 같이 조건부 확률의 곱으로 표현하는 것이 자연스럽다
$$ p(x) = \prod_{i=1}^{n} p(s_n | s_1, ..., s_{n-1})$$
이와 같은 접근은, 다양한 형태의 조건부 확률도 표현할 수 있는데, $p(s_{n-k},...,s_{n} | s_1, ..., s_{n-k-1})$ 와 같은 문장이 나올 확률 같은 것도 모델링이 가능해진다. Transformer와 같은 self-attentive 아키텍쳐는 바로 이러한 조건부 확률을 계산하는 데 있어, 매우 적합하다

또한, 언어 모델의 경우 하나의 모델로 다양한 task에 대해 적용 가능한 특징이 있다.  일반적으로 multi-task learning이나 meta-learning 환경안에서는 구조적 혹은 알고리즘적으로 (MAML 같은) 이러한 부분을 다루는  것이 필요한데, 언어 모델은 이러한 task 정의를 함에 있어, 상대적으로 유연하다는 특징이 있다.  [McCann, 2018](https://arxiv.org/abs/1806.08730)의 연구 결과에 따르면, 각각의 task들은 sequence 형태의 symbol로 표현이 가능한데, 예를 들어
- 번역 예제: (translate tofrench, english text, french text)
- 독해: answer the question, document, question, answer

등과 같이 task가 symbol화 되어 언어모델에 넣을 수 있다. 이는 앞서 언급했듯이 하나의 모델에 다양한 task을 symbol로 입력 받게 되고, 이 를 통해 다양하게 학습을 진행 할 수 있게 되는 것이다.

언어 모델링은 supervised된 출력 결과 없이도 McCann의 task들을 학습할 수 있었는데, 이는 supervised objective와 unsupervised objective가 같기 때문에 이론적으로 실제 전역 최솟값도 동일하게 된다. 즉, unsupervised learning을 통해서도 충분히 큰 언어 모델이 존재하에서 수렴을 가능하도록 할 수 있다는 것이다. (물론 학습 속도면에서는 매우 느릴 것이다.)

### 1. Training dataset
뉴스나 책등과 같이 하나의 도메인의 데이터를 이용한 이전 연구들과 달리 GPT-2에서는 다양한 도메인 데이터를 가져왔다. Common Crawl과 같은 web scrapes는 사실 무제한의 다양한 텍스트를 얻을 수 있는데, 안타깝게도 이중 대부분은 품질이 떨어졌다.

이에 본 연구에서는 고품질의 데이터를 얻기 위해 몇가지 방법을 적용했는데, 다음과 같다
- 사람에 의해 실제 필터링된 web page을 이용
  - Reddit에서 외부링크된 web page을 모아서, 이중 최소한 3 카르마 이상의 글만 사용
  - 4천5백만개 link을 가져옴
- Dragnet과  Newspaper1 content extractors 조합을 통해 텍스트 추출
- 2017/12 이후 링크는 제외했고, 중복제거 및 heuristic한 방법을 통해 데이터를 정제
  - 약 8백만 문서, 40GB의 텍스트 데이터 
  - Wikipedia 데이터 제거: common data source이기에, 학습/테스트 데이터상에 중복 가능성이 있어 제거

### 2. Input representation
범용적으로 사용될 수 있는 언어 모델에서는 어떠한 문장/단어가 입력으로 들어오더라도 이를 처리할 수 있어야 한다. 단어 기반의 언어모델의 경우는 필수적으로 전처리 과정을 포함하게 되는데, 이 과정에서 모델링이 가능한 문자열을 제한하게 된다. 이를 해결하기 위해 Unicode 기반의 언어 모델도 존재하는데, 대규모 데이터 셋에서 평가해보면 단어 기반의 언어모델에 비해 성능이 떨어진다.

이에 본 연구에서는 [__Byte Pair Encoding (BPE)__](https://arxiv.org/abs/1508.07909)을 이용하는데 이는 단어 기반의 접근과 준하는 결과를 얻을 수 있으면서 byte 기반 접근의 장점, 즉 어떠한 문장/단어도 처리 할 수 있게 되었다.

물론 기존의 BPE을 그대로 적용하는 것은 아니다. Byte sequence에 대해 BPE을 그대로 적용하게 될 경우, BPE 알고리즘 자체가 가지고 있는 greedy한 속성에 의해 최적이 아닌게 된다. 예를 들어 dog, dog!, dog?와 같이 다양한 variation 이 BPE에 포함되어 버리는데, 이는 한정된 사전과 모델의 공간을 고려할 때, 피해야 된다. 결국, 문자 범주를 넘어 병합하는 것에 대해 원칙적으로 제한하고, 여러 토큰에 걸쳐 아주 최소한의 부분만 포함되는 경우에 한해서만 예외를 두었다.

이러한 입력 표현은 결국 전처리 과정 없이 어떤 데이터 셋에 대해서도 평가를 진행 할 수 있게 된다. BPE에 관해 자세한 설명은 다음의 [링크](https://wikidocs.net/22592)를 참조 하면 된다.

### 3. Model
모델의 구조는 GPT-1 모델과 거의 유사하다. 몇가지 변경이 있는데 다음과 같다
1. Layer normalization 위치 변경: 각 sub-block의 입력단으로 이동했는데, 이는 pre-activation residual network와 유사하다. 또한 마지막 self-attention block 뒤에도 추가하였다.
1. 모델 깊이에 따른 Residual layer의 초기화 방법이 변경: Residual layer의 갯수 $N$에 따라 가중치 $1/\sqrt{N}$을 적용
1. 사전 갯수: 50,257, Context size: 1024, Batch size: 512

## __Experiment__
본 연구에서는, 4가지 크기의 모델을 제안하고 평가하였다. 가장 작은 모델의 경우 GPT-1과 동등 수준으로 구성되었고, 두번째 작은 모델의 경우는 BERT와 유사 수준이다. 가장 큰 모델의 경우 (이를 앞으로 GPT-2라고 부르자), GPT-1 대비 order 하나 차이가 날 정도로 큰데, 당연히 월등한 성능을 보여준다.

논문에 워낙 많은 task에 대해 성능 평가를 해서 일일히 정리하지 않겠다. 자세한 수치는 논문을 참고 하기 바란다. 

## __Generalization vs Memorization__
최근 컴퓨터 비젼쪽에서 발표된 연구에 의하면, 상당히 많은 데이터 셋에서 동일한 이미지를 볼 수 있다고 한다. CIFAR-10 을 예로 들면, 실제 Train과 Test 데이터셋의 3.3%가 겹친다고 한다. 이러한 중복은 결국 우리의 모델의 일반화를 하는데 있어서 과대 평가를 하는 결과를 낳는다. 

본 연구에서는 8-gram 학습 세트 토큰을 포함하는 Bloom filter을 만들어서 각각의 데이터에 대해 중복 비율을 계산했봤다. 논문에서는 WebText 데이터 셋은 비교적 낮은 겹침을 보여주어서 괜찮다는 결과를 얻었다.

이와 더불어, 기존에도 많이 사용되는 방법이지만, held-out set에 대해 모델 사이즈를 증가시키면서 성능을 비교해 보는 것이다. GPT-2는 모델 사이즈가 증가함에 따라 train/test 데이터 셋 모두에서 유사하게 성능이 향상되는 것을 확인 할 수 있다. 이는 여전히 GPT-2가 over-fitting이 되어 있지 않다는 의미로 memorization 에 의한 성능이 아님을 확인 할 수 있었다.

## __Comment__
Unsupervised train된 대용량 언어 모델을 가지고, zero-shot setting 하에 많은 task에서 우수한 성능을 보여주었다는 점에서 매우 고무적이다. 특히 해당 논문에서는 모델 구조에 대한 새로운 제안보다는, 데이터와 언어모델이 가질 수 있는 특징등에 대해서 더욱 고찰하고, 이를 기반으로 Unsupervised learning 모델의 가능성을 확인 했다는 점에서 개인적으로 재밌게 읽은 논문이였다. 
