---
title:  "[RL] Introduction"
excerpt: "데이비드 실버 교수 자료를 보고 개인적으로 정리한 자료입니다."

categories:
  - Deep Learning
tags:
  - RL
classes: wide
last_modified_at: 2020-12-21T00:21:00+09:00
---

본 포스터는 [데이비드 실버 교수님의 강의](https://www.davidsilver.uk/teaching/)를 공부하면서 정리한 자료이다. 강의에 대해 개인적인 이해를 위해 정리한 것이고 나름의 의견이 가미 되었다. 좀 더 깊고 자세히 공부하고 싶은 분들은 위의 링크를 클릭해 원본 강의자료를 보는 것을 추천한다.

# Introduction
강화학습이 다른 머신러닝 패러다임과 다른 점은 무엇일까?
- Supervisor (정답을 알려주는 사람이)가 없다. Reward가 존재
- 피드백이 즉각적이지 않고, 늦어질 수 있음
- Sequential 하다 (non i.i.d data)
- Agent의 action은 subsequent 데이터에 영향

우선, 강화학습에 사용되는 _용어_ 및 _개념_ 에 대해 알아보자
## Reward 
- Scalar feedback signal
- 현재 시간 t에서 agent가 얼마나 잘 하는지를 나타내는 지표
- Agent는 누적된 reward을 최대화 하는 것을 목적

## Sequential Decision Making
- 목적은 __"미래에 예상되는 Reward 전체 총합을 최대하 하는 action을 선택한다"__ 이다.
  - 왜 현재 Reward에 집중하기 보다 미래에 예상되는 Reward들에 집중할까?
  -  한순간 잘한다고 잘하는 것일까? Action의 결과는 긴 시간에 걸쳐 나타날 수 있고, 또한 그에 따른 Reward가 늦게 나타날 수 있음
- ex) 장기에서 현재 마 같은 기물을 잃지만, 이것이 오히려 발판이 되어 상대의 왕을 잡을 수 있음

## Agent 와 Environment
- Agent 가 어떤 action을 하면, Enviornment는 그에 반응해 Reward을 주고, 뿐만 아니라 agent가 관찰할 수 있는 observation을 전달
- 둘은 상호 작용

## History
- Observations, actions, rewards 들이 차례대로 연결된 sequence: $H_t = O_1, R_1, A_1, ..., A_{t-1}, O_t, R_t$

## State
- State는 history의 function으로 표현: $S_t = f(H_t)$ 
- 다음에 무엇이 일어날지를 결정하는데 사용되는 정보
- 다음의 3가지 state가 정의
### 1. Environment state ($S_t^e$)  
- Environment's private representation
- 일반적으로 agent에 보이지 않음, 설령 보인다 하더라도, 비적절한 정보도 포함할 수 있음
- ex) Atari 게임에서 agent가 한 action을 기준으로 다음 tick에 어떤 화면을 보여줘야 될지 결정하게 되는 게임 내의 정보
### 2. Agent state ($S_t^a$)
- Agent's internal representation
- 다음 action을 하는데 있어 agent가 선택할 수 있는 정보
### 3. Information state 
  - Markov state라고도 불리며, history로 부터 얻을 수 있는 모든 유용한 정보를 포함
  - Markov 정의:
    > A state $S_t$ is _Markov_ iff $P[S_{t+1}|S_{t}] = P[S_{t+1}|S_1, ..., S_t]$
  - 현재가 주어진 상태에서 미래는 과거와는 독립적. 즉, __다음 state는 지금의 state에만 영향을 받음__

## Observable
- Fully observable
  - Agent state = Environment state = Information state
- Partially observable
  - Agent는 간접적으로 environment을 관찰
  - Agent state $\neq$ Environment state
  - Agent는 자신의 state, $S_t^{a}$을 계산해야 함


## Agent의 주요 요소
Agent는 3가지 요소를 가진다
- Policy: Agent의 행동 결정 방식
- Value function: 각각의 action과 state에 따른 스코어
- Model: Environment내에 agent가 표현되는 방식을 결정

### 1. Policy
- Agent의 행동 결정 방식
- State와 action이 매핑
- 두가지 타입의 policy 존재
  - Deteministic policy = $a=\pi (s)$
  - Stocastic policy = $\pi (a|s) = P[A_t = a|S_t = s]$

### 2. Value Function
- 앞으로의 reward 값의 예측치
- 현 state의 좋음/나쁨을 평가하는데 사용
- $v_{\pi}(s) = E_{\pi}[R_{t+1} + \gamma R_{t+2} + \gamma ^2 R_{t+3} + ... | S_t = s]$
- Discount factor $\gamma$ 도입: 먼 미래일수록 reward에 대한 가중치를 낮게 가짐
- Value function은 policy가 정해져야 계산 가능

### 3. Model
- Environemnt가 다음에 어떻게 할지를 예측하는 것
- $\mathcal{P}$(다음 state), $\mathcal{R}$ (다음 reward) 존재
  - $\mathcal{P}_{ss'}^a = P[S_{t+1}=s' | S_t = s, A_t = a]$
  - $\mathcal{R}_{s}^a = P[R_{t+1} | S_t = s, A_t = a]$

## Categorizaing RL agents
Policy, Value function 유무에 따라 3가지 타입으로 정의 가능하다.
- Value Based
  - No policy (implicity) + __Value Function__
- Policy Based
  - __Policy__ + No value function
- Actor Critic
  - __Policy__ + __Value function__

또한 모델 유무에 따라서 다시 2가지로 분류가능하다
- Model Free
  - Policy and/or Value fucntion
  - No model
- Model Based
  - Policy and/or Value fuction
  - __Model__

## Learning and Planning
Sequencial decision making 에 두가지 문제가 정의가능한데,
- Reinforcement learning
  - __Environment는 초기에 모름__
  - Agent는 environment와 상호 작용을 통해 policy을 업데이트
- Planning
  - __Environment model을 안다__
  - Agent는 굳이 Environemnt와 상호작용할 필요없이 모델에 입각해서 계산하고 policy을 업데이트

## Exploration and Exploitation
- Exploration: environment에 대해 좀 더 많은 정보를 찾는 것
- Exploitation: 주어진 정보를 최대한 이용해 reward을 최대화 하는 것

## Prediction and Control
- Prediction: 주어진 policy에서 미래를 평가(예측)
- Control: 최적의 policy을 찾아 미래를 최적화 하는 것