# Paper-Reading-Notes $(universal)$

[TOC]



```markdown
### MainTitle

- [x] **MainTitle** (conf) [[paperswithcode]()]
    - Author et al. "Title"


| 核心在哪? | 精读? 代码? | 关键词? | 亮点? | 笔记时间? |
| --------- | ----------- | ------- | ----- | --------- |
|           |             |         |       |           |

---



+ **背景? 提出了什么问题?**
+ **为了解决此问题提出了什么具体的idea?**
+ **如何从该idea形式化地对问题建模、简化并解决的?**
+ **理论方面证明的定理与推导过程?**
+ **这个任务/解决方法有什么意义?**
+ **对论文的讨论/感想?**
```



### Adaptive Task Sampling (class-pair based)

- [x] **Adaptive Task Sampling (class-pair based)** (ECCV 2020) [[paperswithcode](https://paperswithcode.com/paper/adaptive-task-sampling-for-meta-learning)]
    - Liu et al. "Adaptive Task Sampling for Meta-Learning"


| 核心在哪? | 精读? 代码? | 关键词? | 亮点? | 笔记时间? |
| --------- | ----------- | ------- | ----- | --------- |
|           |             |         |       |           |

---

![image-20201012170700254](assets/image-20201012170700254.png)

While a rich line of work focuses **solely on how to extract meta-knowledge across tasks**, we exploit the complementary problem on **how to generate informative tasks**.

We argue that the randomly sampled tasks could be **sub-optimal and uninformative** (e.g., the task of classifying “dog” from “laptop” is often trivial) to the meta-learner. In this paper, we propose **an adaptive task sampling method** to improve the generalization performance.

![image-20201012163225370](assets/image-20201012163225370.png)

+ **背景? 提出了什么问题?**

  **an episodic training paradigm**.

  A series of few-shot tasks are sampled from meta-training data for **the extraction of transferable knowledge across tasks**, which is then applied to **new** few-shot classification tasks consisting of unseen classes during the meta-testing phase.

  + 问题的提出:

    Despite their noticeable improvements, these meta-learning approaches leverage **uniform sampling** over classes to generate few-shot tasks, which ignores the intrinsic relationships between classes when forming episodes.

    上述方法是 uniform sampling, 这忽略了forming episodes时候类之间的内在联系. 在一些领域中 比如集成学习Adaboost对challenging training examples优先训练后续分类器.

    + 很自然的**提出问题**: Can we perform adaptive task sampling and create more difficult tasks for meta-learning?

    + **难点**: one key challenge in task sampling is to define the difficulty of a task. ![image-20201012162851204](assets/image-20201012162851204.png)

  

  + **Review for Episodic Training:**

    1. In each episode of meta-training, we first **sample $K$ classes $\mathbb{L}^{K} \sim \mathbb{C}_{t r}$.**

    2. Then, we **sample $M$ and $N$ labelled images per class** in $\mathbb{L}^{K}$ to **construct** the support set $\mathbb{S}=\left\{\left(s_{m}, y_{m}\right)_{m}\right\}$ and query set $\mathbb{Q}=\left\{\left(q_{n}, y_{n}\right)_{n}\right\}$, respectively.

       从之前sample的类里面sample出 $\mathbb{S}, \mathbb{Q}$.

    3. The episodic training for few-shot learning **是在query set上最优**, The model is parameterized by $\theta$ and the loss is the negative loglikelihood of the true class of each query sample, 即优化:
       $$
       \ell(\theta)=\underset{(S, Q)}{\mathbb{E}}\left[-\sum_{\left(q_{n}, y_{n}\right) \in Q} \log p_{\theta}\left(y_{n} \mid q_{n}, S\right)\right]
       $$
       $p_{\theta}\left(y_{n} \mid q_{n}, S\right)$ 是在support set上的分类概率.

       注意啊上面的损失是**在 query set 上测的**, 但是训练(后验)是**在support上的.**

       梯度下降 $\Delta \ell(\theta)$.

    

  + **Review for Instance-base Adaptive Sampling for SGD:**

    **Select Sample 的概率:**

    + 第一次:
      $$
      p_{0}(i \mid \mathbb{D})=\frac{1}{|\mathbb{D}|}
      $$

    + 之后:

      instance $i$ at iteration $t+1$ **according to the current prediction probability $p\left(y_{i} \mid x_{i}\right)$** and the selection probability at previous iteration $p^{t}(i)$
      $$
      p^{t+1}(i) \propto\left(p^{t}(i)\right)^{\tau} e^{\alpha\left(1-p\left(y_{i} \mid x_{i}\right)\right)}
      $$
      where the hyperparameters $\tau$ is a discounting parameter and $\alpha$ scales the influence of current prediction.

      This multiplicative update method has a close relation to maximum loss minimization [47] and AdaBoost $[16]$.

+ **为了解决此问题提出了什么具体的idea?**

  a straightforward extension of the instance-based sampling.

+ **如何从该idea形式化地对问题建模、简化并解决的?**

  + **Class-based Sampling**:

    We propose a class-based sampling (c-sampling) approach that **updates the class selection probability $p_{C}^{t+1}(c)$ in each episode.**

    具体选择类概率的更新公式如下:

    Given $\mathbb{S}^{t}$ and $\mathbb{Q}^{t}$ at episode $t,$ we could update the class selection probability for each class in current episode $c \in \mathbb{L}_{K}^{t}$ in the following way,

    ![image-20201012171737308](assets/image-20201012171737308.png)

    Note that we average the prediction probability of classifying each query sample $n$ into incorrect classes in $\mathbb{L}_{K}^{t} .$ Then we can sample $K$ classes without replacement to construct the category set $\mathbb{L}_{K}^{t+1}$ for the next episode.

    

    每个类的难度不是独立的.

    取出类别二元组, 无向概率图模型 马尔可夫随机场, 这里不是最大团.

    更新$C (i, j)$, 该类别对在上一次就混淆了, 接下来就要挑这个.

    不能接受的计算复杂度, 则使用贪心算法.

+ **理论方面证明的定理与推导过程?**

+ **这个任务/解决方法有什么意义?**

+ **对论文的讨论/感想?**



Stochastic optimization with importance sampling for regularized loss minimization. (ICML 2015)