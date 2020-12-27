---
title:  "[RL] Markov Decision Process"
excerpt: "데이비드 실버 교수 자료를 보고 개인적으로 정리한 자료입니다."

categories:
  - Deep Learning
tags:
  - RL
classes: wide
last_modified_at: 2020-12-27T00:21:00+09:00
---
본 포스터는 [데이비드 실버 교수님의 강의](https://www.davidsilver.uk/teaching/)를 공부하면서 정리한 자료이다. 강의에 대해 개인적인 이해를 위해 정리한 것이고 나름의 의견이 가미 되었다. 좀 더 깊고 자세히 공부하고 싶은 분들은 위의 링크를 클릭해 원본 강의자료를 보는 것을 추천한다.

# Markov Decision Process (MDP)
RL에서 Environment을 표현하는 방법으로 거의 모든 RL 문제들은 MDP로 수식화 가능하다. 
## 1. Markov Process (MP)
Markov process (__memoryless random process__) 는 다음과 같이 정의된다.
> A _Markov Process_ (or Markov Chain) is a tuple <$\mathcal{S}$, $\mathcal{P}$>  
> - $\mathcal{S}$ is a (finite) set of states
> - $\mathcal{P}$ is a state transition probability matrix,  
 $\mathcal{P}_{ss'}=P[S_{t+1} = s'|S_t = s]$

## 2. Markov Reward Process (MRP)
Markov Reward Process는 다음과 같이 정의된다.
> A _Markov Reward Process_ is a tuple <$\mathcal{S}$, $\mathcal{P}$, $\mathcal{R}$, $\gamma$>  
> - $\mathcal{S}$ is a (finite) set of states
> - $\mathcal{P}$ is a state transition probability matrix,  
 $\mathcal{P}_{ss'}=P[S_{t+1} = s'|S_t = s]$
> - __$\mathcal{R}$ is a reward function, $\mathcal{R}_s = E[R_{t+1} | S_t = s]$__
> - __$\gamma$ is a discount factor, $\gamma \in  [0, 1]$__

Return이 새로 정의되는데, 다음과 같다
> The return $G_t$ is the total discounted reward from time-step $t$,
> $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma ^2 R_{t+3} + ... = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}$$

여기서 Discount을 사용하는데, why?
- 수학적으로 사용이 편리(convergence)
- Future의 uncertainty을 표현
- 금전적인 경우 예로 들면, 현재의 보상이 훨씬 좋음 (이자 개념)  
  혹은, 행동학적 관점에서 동물/사람의 행동은 즉각적인 보상에 선호도가 보임
- 그럼 $\gamma=1$인 경우? 만약 모든 sequence가 무조건 종료가 사용 가능

MRP에서는 value function이 다음과 같이 정의 된다.
> The state value function $v(s)$ of an MRP is the expected return
starting from state $s$,
> $$v(s) = E [G_t| S_t = s]$$

Value function은 확률로써 표현 가능하고, 즉, 우리는 다양한 샘플 state sequence을 이용해 value function을 계산 가능하다. 하지만 다른 방법이 있다는데????


