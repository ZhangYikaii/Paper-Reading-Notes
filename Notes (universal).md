# Paper-Reading-Notes $(universal)$

[TOC]



三个维度: 数据集, x-way y-shot, 网络/方法.

## `miniimagenet`:

| Method ↓ / Setups → | 5-Way 5-Shot                                              | 5-Way 1-Shot |
| ------------------- | --------------------------------------------------------- | ------------ |
| Proto               | 61.8 (`miniImageNet_nt=5_kt=20_qt=15_nv=5_kv=5_qv=1.csv`) |              |
| Matching            |                                                           |              |
|                     |                                                           |              |



```markdown
### MainTitle

- [x] (conf) [[paperswithcode]()]
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



> Diversity with Cooperation: Ensemble Methods for Few-Shot Classification
>
> Transferable Meta Learning Across Domains
>
> A Closer Look at Few-shot Classification
>
> Boosting Few-shot Image Recognition via Domain Alignment Prototypical Networks
>
> Cross-domain Few-shot Classification via Learned Feature-wise Transformation



### CloserLookFewShot

- [x]  (ICLR 2019) [[paperswithcode](https://paperswithcode.com/paper/a-closer-look-at-few-shot-classification-1)]
    - Chen et al. "A Closer Look at Few-shot Classification"


| 核心在哪? | 精读? 代码? | 关键词? | 亮点? | 笔记时间? |
| --------- | ----------- | ------- | ----- | --------- |
|           |             |         |       |           |

---

![image-20201022182037276](assets/image-20201022182037276.png)

![image-20201022174829109](assets/image-20201022174829109.png) ![image-20201022181334771](assets/image-20201022181334771.png)

reducing intra-class variation 是重要的: when the feature backbone is shallow.

+ **背景? 提出了什么问题?**

  在一个更加公平的基础上来比较各个Few-Shot Learning算法之间的优劣.

  在研究 复现和实验现有算法过程中, 总结出的一份实验报告.

+ **为了解决此问题提出了什么具体的idea?**

  + 提出 consistent **comparative experiments** to compare several representative few-shot classification methods on common ground.

    在设置基类和新类的有限域差异时, 使用一个deep backbone可以缩小不同方法之间的性能差距.

  + 在distance-based classifier中替换线性分类器, 可以达到SotA.

  + 提出了一个practical evaluation setting. 其中存在基类和新类之间的领域转换.

+ **如何从该idea形式化地对问题建模、简化并解决的?**

  较多的base class的标记数据: $\mathbf{X}_{b}$, 较少的novel class标记数据 $\mathbf{X}_{n}$.

  在训练过程中 $\mathbf{X}_{n}$ 是unseen的.

  + 训练过程: 交叉熵损失 分类器 $C\left(. \mid \mathbf{W}_{b}\right)$ 由 $\mathbf{W}_{b}^{\top} f_{\theta}\left(\mathbf{x}_{i}\right)$ 的线性层, 然后是softmax函数.
  + Fine-tuning 过程: adapt 模型去识别新类, 固定feature extractor $f_\theta$ 的参数, 使用交叉熵损失训练一个新的分类器 $C\left(. \mid \mathbf{W}_{n}\right)$.

  **基于baseline的提升:**

  在训练时 explicitly reduces intra-class variation among features.

  ![image-20201023171643802](assets/image-20201023171643802.png)

  

+ **理论方面证明的定理与推导过程?**

+ **这个任务/解决方法有什么意义?**

+ **对论文的讨论/感想?**





### MainTitle

- [ ]  (ICLR 2020) [[paperswithcode](https://paperswithcode.com/paper/cross-domain-few-shot-classification-via-1)]
  - Tseng et al. "Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation"


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



## About Attention/GNN

+ 近期:

  Attentive Graph Neural Networks for Few-Shot Learning

  Few-Shot Learning With Attention-Weighted Graph Convolutional Networks For Hyperspectral Image Classification

  

+ 相关:

  A Dual Attention Network with Semantic Embedding for Few-shot Learning

  **Multi-attention Meta Learning for Few-shot Fine-grained Image Recognition**

  **Attentive Weights Generation for Few Shot Learning via Information Maximization**

  

+ 高引:

  Dynamic Few-Shot Visual Learning Without Forgetting

  Few-Shot Learning with Graph Neural Networks

  TADAM

  **Cross Attention Network for Few-shot Classification**

  Few-Shot Semantic Segmentation with Prototype Learning





### FewShotWithoutForgetting

- [x] (CVPR 2018) [[paperswithcode](https://paperswithcode.com/paper/dynamic-few-shot-visual-learning-without)]
    - Gidaris et al. "Dynamic Few-Shot Visual Learning without Forgetting"


| 核心在哪? | 精读? 代码? | 关键词? | 亮点? | 笔记时间? |
| --------- | ----------- | ------- | ----- | --------- |
|           |             |         |       |           |

---

基于外部记忆的. base类和novel类之间是没有相同的. weight generator还把base weight作为输入. 先训练出feature extractor固定, 再训练后面的weight generator.

一般方法的不足/我们期望得到的.

+ 基类上的分类能力不能降, 新类上的学习要快, 效果要好.

+ ConvNet-based recognition model:
  $$
  p = C\left( F \left(x \mid \theta \right) \mid W_{b a s e} \right)
  $$

  + 令 $z = F \left(x \mid \theta \right)$, 我们有:
    $$
    p=\tau \cdot \cos \left(z, \ w\right)=\tau \cdot \bar{z}^{\top} \bar{w}
    $$

+ Few-shot classification weight generator:
  $$
  w^{\prime}=G\left(F \left(x^{\prime} \mid \theta \right), W_{b a s e} \mid \phi\right)
  $$

  + 如下, $\phi$ 都为learnable weight matrix; 分为$w_{a v g}^{\prime}, w_{a t t}^{\prime}, w^{\prime}$三个部分:
    $$
    \begin{aligned}
    w_{a v g}^{\prime}& = \frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}} \bar{z}_{i}^{\prime} \\
    w_{a t t}^{\prime}& = \frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}} \sum_{b=1}^{K_{b a s e}} A t t\left(\phi_{q} \bar{z}_{i}^{\prime}, k_{b}\right) \cdot \bar{w}_{b} \\
    w^{\prime} & =\phi_{a v g} \odot w_{a v g}^{\prime}+\phi_{a t t} \odot w_{a t t}^{\prime}
    \end{aligned}
    $$

+ 训练阶段 **1** 学习: $C$ 的 $\theta, \ W_{base}$.

  训练阶段 **2** 学习: 采样得fake novel class的$z$, (请注意 $W_{base}$ 蕴含在此), 喂给$G$学习参数 $\phi$, 训练得到的 $w'$ ($G$的输出)作为真正novel class的分类器参数.

注意力在注意啥? 代码中如何拆解为一般形式并理解?

+ **背景? 提出了什么问题?**

  需要做到:

  (a) the learning of the **novel categories needs to be fast**, 在新类上的学习要快.

  (b) to not **sacrifice** any recognition accuracy on **the initial categories** that the ConvNet was trained on, i.e., to not “forget” (from now on we will refer to those initial categories by calling them base categories). 原来的类上的性能不能降.

  not only is able to **recognize these base categories**, but also learns to **dynamically** recognize novel categories from only a few training examples (provided only at test time) while also not forgetting the base ones or requiring to be re-trained on them (dynamic few-shot learning without forgetting).

  不会忘记base class, 也不需要re-trained.

+ **为了解决此问题提出了什么具体的idea?**

  + **Few-shot classification-weight generator based on attention.**

    the first technical novelty of our work is that we enhance a typical object recognition system with an extra component, called **few-shot classification weight generator**:

    1. input: **a few training examples** of a novel category e.g., no more than five examples) and, 
    2. based on them, generates **a classification weight vector** for that novel category.

    + acquired **past knowledge** about the visual world:

      通过 incorporating an attention mechanism **over the classification weight vectors** of the base categories.

      注意力机制作用在base categories上.

    

  + **Cosine-similarity based ConvNet recognition model.**

    ConvNet 需要可以处理基类和新类的classification weight vectors.

    问题是: However, as we will explain in the methodology, this is **not** feasible with the **typical dot-product based classifier** (i.e., the last linear layer of a classification neural network). 

    implement the classifier as a **cosine similarity function** between **the feature representations and the classification weight vectors.**

    ![image-20201023212202447](assets/image-20201023212202447.png)



+ **如何从该idea形式化地对问题建模、简化并解决的?**

  + **3.Methodology**

    base categories: $K_{base}$

    第 $b$ 类的训练样本数量: $N_b$, 第 $b$ 类的第 $i$ 个: $x_{b, i}$.

    ![image-20201023214207921](assets/image-20201023214207921.png)

    接下来, 如上两个部分: a ConvNet-based recognition model that is able to recognize both base and novel categories, a few-shot classification weight generator: dynamically generates **classification weight vectors** for the novel categories at test time.

    + **a ConvNet-based recognition model**:

      feature extractor: $F(.\mid \theta), z=F(x \mid \theta) \in \mathbb{R}^{d}$.

      classifier: $C(.\mid W^*), p=C\left(z \mid W^{*}\right)$, **the probability classification scores $p$.**

      这个阶段学到了:

      + classification weight vectors: $W^*$
      + $\theta$

      

    + **Few-shot classification weight generator**:
      $$
      w_{n}^{\prime}=G\left(Z_{n}^{\prime}, W_{b a s e} \mid \phi\right)
      $$

  + **3.1.Cosine-similarity based recognition model**

    计算第$k$类的score:
    $$
    s_{k}=z^{\top} w_{k}^{*}
    $$
    请注意: However, **the mechanisms involved during learning those classification weights $w_k^*$ are very different.**

    + 对于base类的权值, 学习过程是缓慢的.
    + 对于novel类的权值, 学习是快速的, 因为基于输入特征.

    Due to those differences, the weight values in those two cases (i.e., base and novel classification weights) **can be completely different,** and so the same applies to the raw classification scores computed with the dot-product operation, which can **thus have totally different magnitudes** depending on whether they come **from the base or the novel categories.**

    base 类权重一样训练得到, novel 类的权重是weight generator的输出. 这样分类器就同时具备对基础类别和新类别进行分类的能力.

    因为这两种方式得到的权重有较大差异(**量级问题**), 传统的点积方法计算相似度会造成较大误差:

    所以修改 分类器$C(.\mid W^*)$为计算cos相似度:
    $$
    s_{k}=\tau \cdot \cos \left(z, w_{k}^{*}\right)=\tau \cdot \bar{z}^{\mathrm{T}} \bar{w}_{k}^{*}
    $$
    where $\bar{z}=\frac{z}{\|z\|}$ and $\bar{w}_{k}^{*}=\frac{w_{k}^{*}}{\left\|w_{k}^{*}\right\|}$ are the $l_{2}$ -normalized vectors (from now on we will use the overline symbol $\bar{z}$ to indicate that a vector $z$ is $l_{2}$ normalized , and $\tau$ is a learnable scalar value. 消除了基础类别权重和新类别权重在数值上的量级差异.

    

    **Advantages of cosine-similarity based classifier, 请再看原文, 作者为此还做了一个可视化.**

    

  + **3.2.Few-shot classification weight generator**

    $w^{\prime}=G\left(Z^{\prime}, W_{b a s e} \mid \phi\right)$ classification weight generator 的构建方式:

    **1 Feature averaging based ($\bar{z}_{i}^{\prime}$ 是特征向量)**
    $$
    w^{\prime}=\phi_{a v g} \odot w_{a v g}^{\prime} = \phi_{a v g} \odot \left(\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}} \bar{z}_{i}^{\prime} \right)
    $$
    **2 Attention-based**

    上述平均的方法不好, 特别是新类别只包含一个样本(one-shot情形)时不能充分利用特征提取网络的信息.
    $$
    w_{a t t}^{\prime}=\frac{1}{N^{\prime}} \sum_{i=1}^{N^{\prime}} \sum_{b=1}^{K_{b a s e}} A t t\left(\phi_{q} \bar{z}_{i}^{\prime}, k_{b}\right) \cdot \bar{w}_{b}
    $$

    $$
    w^{\prime}=\phi_{a v g} \odot w_{a v g}^{\prime}+\phi_{a t t} \odot w_{a t t}^{\prime}
    $$

    

+ **理论方面证明的定理与推导过程?**

+ **这个任务/解决方法有什么意义?**

+ **对论文的讨论/感想?**



### CAN

- [x] (NeurIPS 2019) [[paperswithcode](https://paperswithcode.com/paper/cross-attention-network-for-few-shot)]
    - Hou et al. "Cross Attention Network for Few-shot Classification"


| 核心在哪? | 精读? 代码? | 关键词? | 亮点? | 笔记时间? |
| --------- | ----------- | ------- | ----- | --------- |
|           |             |         |       |           |

---

we propose a novel Cross Attention Network to address the challenging problems in few-shot classification.

Firstly, Cross Attention Module is introduced to deal with the problem **of unseen classes**. The module generates **cross attention maps** for each pair of class feature and query sample feature so as to highlight the target object regions, making **the extracted feature more discriminative.** Secondly, a transductive inference algorithm is proposed to **alleviate the low-data problem**, which iteratively utilizes the unlabeled query set to augment the support set, thereby making the class features more representative.

+ **背景? 提出了什么问题?**

  难点(通用的):

  the test images in the support/query set are **from unseen classes, thus their features can hardly** attend to the target objects.

  ![image-20201024154007518](assets/image-20201024154007518.png)

  the extracted features only capture the information of the objects that are **related to the training classes**.

  For another, the low-data problem makes the feature of each test class **not representative** for the true class distribution, as it is obtained from very few labeled support samples.

  **enhance the feature discriminability for few-shot classification.**

  To recognize a sample from unseen class given a few labeled samples, human tends to **firstly locate the most relevant** regions in the pair of labeled and unlabeled samples.

  **Transductive Algorithm 未看.**

+ **为了解决此问题提出了什么具体的idea?**

  

+ **如何从该idea形式化地对问题建模、简化并解决的?**

  metric-learning to obtain proper feature representations

  we propose Cross Attention Module (CAM): model **the semantic relevance between the class feature and query feature**, thus draw attention to the target objects and benefit the subsequent matching.

  ![image-20201024161842119](assets/image-20201024161842119.png)

  $P^{k}$ 代表第$k$类的特征图, $Q^{b}$ 代表一个query样本的特征图.

  Correlation Layer 计算$P, Q$之间的correlation map. 对应点利用cos距离计算相关性.

  

+ **理论方面证明的定理与推导过程?**

+ **这个任务/解决方法有什么意义?**

+ **对论文的讨论/感想?**

  the episode training mechanism: The episodes used in training **simulate the settings in test**. the settings in test. Each episode is formed by **randomly sampling $C$ classes and $K$ labeled samples per class** as the support set $\mathcal{S}=\left\{\left(x_{a}^{s}, y_{a}^{s}\right)\right\}_{a=1}^{n_{s}}\left(n_{s}=C \times K\right),$ and a fraction of the rest samples from the $C$ classes as the query set $\mathcal{Q}=\left\{\left(x_{b}^{q}, y_{b}^{q}\right)\right\}_{b=1}^{n_{q}} .$ And we denote $\mathcal{S}^{k}$ as the support subset of the $k^{t h}$ class. **How to represent each support class $\mathcal{S}^{k}$ and query sample $x_{b}^{q}$ and measure the similarity between them** is a key issue for few-shot classification.



### Graph Similarity

+ GNN: 同时感知图结构和feature.

  aggregate: 用邻居的feature更新下一层的hidden state. 采用卷积(1. Spatial-based, 2. Spectral-based)

  readout: 所有nodes的feature集合起来代表这个图.

  第 $l$ 层图卷积写成这样的非线性函数: $H^{l+1}=f\left(H^{l}, A\right)$, $H$ 是上一层输出, $H^0$ 是节点的特征矩阵.

  离散卷积本质就是一种加权求和, CNN中的卷积本质上就是利用一个共享参数的kernel, 通过计算中心像素点以及相邻像素点的加权和来构成feature map实现空间特征的提取. 加权参数(kernel的)是学习得到的.

  Spectral graph theory(有wiki): 借助于图的拉普拉斯矩阵的特征值和特征向量 来研究图的性质.

  定义Laplacian算子的目的是为了找到Fourier变换的基. 图上=>拉普拉斯矩阵. 随即定义图上Fourier变换.
  
+ 图 相似度:

  kernel method:

  1 Graph embedding 算法, 将图(Graph)结构嵌入到向量空间, 得到图结构的向量化表示, 直接应用基于向量的核函数(RBF kernel, Sigmoid kernel, etc.) 处理, 但是这样降维损失了大量结构化信息; 

  2 Graph kernel, 直接面向图结构数据, 核函数计算高效的优点, 又包含了图数据在希尔伯特高维空间的结构化信息.

+ Contrastive Methods只需要在特征空间上学习到区分性. 不会过分关注像素细节, 而能够关注抽象的语义信息. 一般范式: 学到与相似样本的相似度得分更高, 不相似样本的得分低. 表示学习算法并不一定要关注到样本的每一个细节, 只要学到的特征能够使其和其他样本区别开来就行.

  motivation: 深度学习是1 Representation Learning, 2 Inductive Bias Learning, 在一些不涉及逻辑、推理的问题上表现较好. InfoNCE.

  自监督: 任意挖掘对象之间联系、探索不同对象共同本质的方法.

- [x] (NIPS 2020) [[paperswithcode]()]
    - Author et al. "A Graph Similarity for Deep Learning"


| 核心在哪? | 精读? 代码? | 关键词? | 亮点? | 笔记时间? |
| --------- | ----------- | ------- | ----- | --------- |
|           |             |         |       |           |

---

GNN 聚合-transform, 聚合邻居的属性然后通过可学习的函数transform. 我们缺乏对这些representation有多相似的.

采用kernel距离, 提出 transform-sum-cat, 在邻域聚合中, 通过聚合-transform来反映节点邻域之间的连续相似性. 进而引出Weisfeiler–Leman similarity (WLS), 与现有的图kernel相比, WLS很容易用普通的深度学习框架实现. 
在图分类实验中, transform-sum-cat方法的性能明显优于目前流行的GNN模型中的其他邻域聚集方法.

+ **背景? 提出了什么问题?**

  从前图的相似性度量: Weisfeiler–Leman (WL) algorithm. 该算法核心是做 neighborhood aggregation.

  问题提出: 连续属性在邻域聚合时发送问题. 提出 transform-sum-cat, 其中cat表示与来自中心节点的信息的连接, 并应用到WL算法上.

  ![image-20201127113628054](assets/image-20201127113628054.png)

  + WL算法 同构测试, 在有限步后比较: ![image-20201127114555019](assets/image-20201127114555019.png)

  + the kernel distance between the point sets:
    
    $K$ 是一个正定核, 相当于有个映射 $\phi$ 到希尔伯特空间.
  
  

  先aggregate, 再transform, 这个顺序没有可靠理论证明. 先aggregate在一些例子中可能带来危险: 不同的图但是 一样.

+ **为了解决此问题提出了什么具体的idea?**

  一个好的neighborhood representation:

  1 旋转不变, 2 平移不变, 3 单调性.

+ **如何从该idea形式化地对问题建模、简化并解决的?**

  Weisfeiler–Leman similarity: **reflect** the similarity **between the sets of neighbors’ attributes** into the **node-wise updated** attributes via the set-representation vector from Section 2.3.
  
  ![image-20201128105013323](assets/image-20201128105013323.png)
  
  单个迭代步骤对应于 transform-sum-cat.
  
  注意是先计算希尔伯特矩阵, 就是每两个图之间的距离 $\mathcal{K} (G_1, G_2)$, 搞好之后喂到svm里面, 直接一个度量的矩阵, 可以不需要知道 $\phi (G_1)$.


+ **理论方面证明的定理与推导过程?**
+ **这个任务/解决方法有什么意义?**
+ **对论文的讨论/感想?**



## Contrastive Learning

+ 图上的, (关于 Self-supervised Learning: Generative or Contrastive):

  三类: Generative、Contrastive、Adversarial(Generative-Contrastive)

  
  
  



+ 不同领域/任务之间 有共享的学习器, 有刻画任务不同的学习器.

  共享了一部分参数, 假设空间 $\mathcal{H}$ 的大小会更小.

  理论层面 $\mathcal{O} (\frac{1}{\sqrt{NT}} )$, $N$ 是任务的数目. 多模态是多任务的一个特例. few-shot 即在$N$很大时是可行的.

  多任务的应用是广的, 比如多loss训练.

+ 多任务的setting(看陆学长图): 学到每个任务/数据集对应的分类器, 每个数据集有一个对应的测试集进行测试.

  meta learning: 将原有任务数据集上扩展到新的任务上. 期望跨任务拓展.

  所有任务都是从共同的分布中抽取的.

+ few-shot NFL 不可能直接在样本很小上学习出. 从已有的知识/数据/类别上构造.

  eg: 目标做 2-way, 1-shot, **则这样构造训练集**:

  抽样, 例如凑出了A, B两个类别, 又抽出了 $a_1, b_1$ 两个样本, 这就是一个任务.

  同理又抽出一个任务 $b_2, d_2$. 如此可以构造非常多的任务. 通过这样一个过程去测试新的少样本任务. 同样测试集就是抽完类别之后抽一个和测试集不重叠的.

  注意上面都是拿去训练的, 都是带标签的, 上面抽的就是support和query.

  多任务评估:
  $$
  \min \frac{1}{T} \sum_t \min_n \frac{1}{N} \sum_i \mathscr{L} ( h \cdot g (x_i), y)
  $$
  few-shot评估见手机拍照. 很像但是有点不同, query set中的元素依赖support中得到的预测和query set的标签算loss.

  已有数据的模拟, 达到对未知数据的预测能力.

  因为这样在原样本/训练样本上query是算loss的, 所以我们任务泛化到少样本是可以解释的.

  如何做好内层优化: 1 最近邻, query 和 support 之间的.

  最近邻方法和similarity相关, 一般有 1 cosine (Matching Network), 2 计算出每个类的中心, 和类别中心算相似度.

  MAML 固定步数的梯度下降来逼近这个点. 小样本学习是一个问题, 这个问题也可以用其他方法解比如迁移学习等. 元学习就是一类方法.

  期望新任务上的模型与理想模型非常近似, 但是在模型空间上比较模型的近似是很难的(模型参数多, 且理想模型不好训练), 所以转而在样本空间上比较, 即 $f (D_{tr})$ 这个模型对 $x$ 的预测是非常准的, 即非常像理想模型了.

  构造训练集非常关键, 这和long-tail或主动学习等都有很大关系. 元学习: 通过目标模型调整抽样方式. 元学习形而上学, 样本空间往上走一层.

