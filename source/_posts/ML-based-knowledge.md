---
title: ML-based-knowledge
categories: ML
abbrlink: '96138251'
date: 2024-09-25 16:58:41
updated: 2024-09-25 16:59:41
tags:
mathjax: true
toc: true
---
<meta name="referrer" content="no-referrer"/>

# ML知识点汇总

## 1.LSTM

1. [原理分析](https://blog.csdn.net/qq_38147421/article/details/107692418)

## 2.预训练思想

有了图像领域预训练的引入，我们在此给出预训练的思想：任务 A 对应的模型 A 的参数不再是随机初始化的，而是通过任务 B 进行预先训练得到模型 B，然后利用模型 B 的参数对模型 A 进行初始化，再通过任务 A 的数据对模型 A 进行训练。注：模型 B 的参数是随机初始化的。

## 3.神经网络语言模型NNLM

神经网络语言模型则引入神经网络架构来估计单词的分布，**并且通过词向量的距离衡量单词之间的相似度，因此，对于未登录单词，也可以通过相似词进行估计，进而避免出现数据稀疏问题**。

![image-20240925143535796](https://img-blog.csdnimg.cn/direct/4063e81a2a4f47569b2231ca4e19822a.png)

上图所示有一个 $V×m $ 的矩阵 $Q$，这个矩阵$Q$包含 $V$ 行，$V$ 代表词典大小，每一行的内容代表对应单词的 Word Embedding 值。

只不过  $Q$ 的内容也是网络参数，需要学习获得，训练刚开始用随机值初始化矩阵  $Q$，当这个网络训练好之后，矩阵  $Q$ 的内容被正确赋值，每一行代表一个单词对应的 Word embedding 值



上图为神经网络语言模型结构图，它的学习任务是输入某个句中单词 $w_t$=bert 前的 t−1 个单词，要求网络正确预测单词 “bert”，即最大化：

$P(w_t=bert|w1,w2,⋯,wt−1;θ)$

上图所示的神经网络语言模型分为三层，接下来我们详细讲解这三层的作用：

1. 神经网络语言模型的第一层，为输入层。首先将前 n−1 个单词用 Onehot 编码（例如：0001000）作为原始单词输入，之后乘以一个随机初始化的矩阵 Q 后获得词向量 $C(w_i)$，对这 n−1个词向量处理后得到输入 x，记作 $x=(C(w_1),C(w_2),⋯,C(w_{t−1}))$
2. 神经网络语言模型的第二层，为隐层，包含 h 个隐变量，H 代表权重矩阵，因此隐层的输出为 $H_x+ d$，其中 d 为偏置项。并且在此之后使用 tanh 作为激活函数。
3. 神经网络语言模型的第三层，为输出层，一共有 $|V|$ 个输出节点（字典大小），直观上讲，每个输出节点$yi$是词典中每一个单词概率值。最终得到的计算公式为：$y=softmax(b+W_x+Utanh⁡(d+H_x))$，其中 W 是直接从输入层到输出层的权重矩阵，U 是隐层到输出层的参数矩阵。

* Word Embedding 其实就是**标准的预训练过程**

## 4.词向量

### 独热编码

**把单词用向量表示，是把深度神经网络语言模型引入自然语言处理领域的一个核心技术。**

在自然语言处理任务中，训练集大多为一个字或者一个词，把他们转化为计算机适合处理的数值类数据非常重要。

早期，人们想到的方法是使用独热（Onehot）编码，如下图所示![image-20240925145525155](https://img-blog.csdnimg.cn/direct/5b002586caa9442ca55946e3e5433dab.png)

但是，对于独热表示的向量，如果采用余弦相似度计算向量间的相似度，**可以明显的发现任意两者向量的相似度结果都为 0**，即任意二者都不相关，也就是说独热表示无法解决词之间的相似性问题

### Word Embedding

在神经网络语言模型中出现的一个词向量 C(wi)，对的，**这个 C(wi) 其实就是单词对应的 Word Embedding 值，也就是我们这节的核心——词向量。**

![image-20240925143535796](https://img-blog.csdnimg.cn/direct/4063e81a2a4f47569b2231ca4e19822a.png)

上图所示有一个 $V×m $ 的矩阵 $Q$，这个矩阵$Q$包含 $V$ 行，$V$ 代表词典大小，每一行的内容代表对应单词的 Word Embedding 值。

只不过  $Q$ 的内容也是网络参数，需要学习获得，训练刚开始用随机值初始化矩阵  $Q$，当这个网络训练好之后，矩阵  $Q$ 的内容被正确赋值，每一行代表一个单词对应的 Word embedding 值

但是这个词向量有没有解决词之间的相似度问题呢？为了回答这个问题，我们可以看看词向量的计算过程：

$[0&0&0&1&0] \begin{matrix}7&24&1\\23&5&7\\4&6&13\\10&12&19&\\11&18&25 \end{matrix} = [10&12&19]$



通过上述词向量的计算，可以发现第 4 个词的词向量表示为 [10 12 19]。

如果再次采用**余弦相似度计算两个词之间的相似度，结果不再是 0** ，既可以一定程度上描述两个词之间的相似度

### Word2Vec模型

* Word2Vec工作原理

![image-20240925151659106](https://img-blog.csdnimg.cn/direct/e3ce8be556be47b596df3d82c1f4a734.png)

Word2Vec 的网络结构其实和神经网络语言模型（NNLM）是基本类似的，不过这里需要指出：尽管网络结构相近，而且都是做语言模型任务，但是**他们训练方法不太一样**。

Word2Vec 有两种训练方法：

1. 第一种叫 **CBOW**，**核心思想是从一个句子里面把一个词抠掉**，用这个词的上文和下文去预测被抠掉的这个词；
2. 第二种叫做 **Skip-gram**，和 CBOW 正好反过来，输入某个单词，要求网络预测它的上下文单词。

而NNLM的训练方法是**输入一个单词的上文，去预测这个单词**

为什么 Word2Vec 这么处理？原因很简单，因为 Word2Vec 和 NNLM 不一样，NNLM 的主要任务是要学习一个解决语言模型任务的网络结构，语言模型就是要看到上文预测下文，而 Word Embedding只是 NNLM 无心插柳的一个副产品；但是 Word2Vec 目标不一样，它单纯就是要 Word Embedding 的，这是主产品，所以它完全可以随性地这么去训练网络。

### EMLO

word embedding无法区分多义词。

ELMo 的本质思想是：先用语言模型学好一个单词的 Word Embedding，此时多义词无法区分，不过这没关系。在实际使用 Word Embedding 的时候，单词已经具备了特定的上下文了，这个时候我可以根据上下文单词的语义再去调整单词的 Word Embedding 表示，这样经过调整后的 Word Embedding 更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以 ELMo 本身是个**根据当前上下文对 Word Embedding 动态调整的思路。**

ELMo 采用了典型的两阶段过程：

1. 第一个阶段是利用语言模型进行**预训练**；
2. 第二个阶段是在做**下游任务**时**，从预训练网络中提取对应单词的网络各层的 Word Embedding 作为新特征补充到下游任务中。**

![image-20240925164420747](https://img-blog.csdnimg.cn/direct/5aa3f88d24e546edac02bcabf59f17e7.png)

上图展示的是其**第一阶段预训练过程**，它的网络结构采用了双层双向 LSTM，目前语言模型训练的任务目标是根据单词 $w_i$ 的上下文去正确预测单词 $w_i$ ，$w_i$ 之前的单词序列 Context-before 称为上文，之后的单词序列 Context-after 称为下文。

图中左端的前向双层 LSTM 代表正方向编码器，输入的是从左到右顺序的除了预测单词外$w_i$的上文 Context-before；右端的逆向双层 LSTM 代表反方向编码器，输入的是从右到左的逆序的句子下文Context-after；每个编码器的深度都是两层 LSTM 叠加。

使用这个网络结构利用大量语料做语言模型任务就能预先训练好这个网络，如果训练好这个网络后，输入一个新句子 $s_{new}$ ，句子中每个单词都能得到对应的三个 Embedding：

- 最底层是单词的 Word Embedding；
- 往上走是第一层双向 LSTM 中对应单词位置的 Embedding，这层编码单词的**句法信息**更多一些；
- 再往上走是第二层 LSTM 中对应单词位置的 Embedding，这层编码单词的**语义信息**更多一些

也就是说，ELMo 的预训练过程不仅仅学会单词的 Word Embedding，还学会了一个双层双向的 LSTM 网络结构，而这两者后面都有用。

#### ELMo 的 Feature-based Pre-Training

预训练好之后，elmo如何给下游任务使用呢？

![image-20240925165426685](https://img-blog.csdnimg.cn/direct/be676db332e64faba09f2333dd5e1b42.png)

上图展示了下游任务的使用过程，比如我们的下游任务仍然是 QA 问题，此时对于问句 X：

1. 我们可以先将句子 X 作为预训练好的 ELMo 网络的输入，这样句子 X 中每个单词在 ELMO 网络中都能获得对应的三个 Embedding；
2. 之后给予这三个 Embedding 中的每一个 Embedding 一个权重 a，这个权重可以学习得来，根据各自权重累加求和，将三个 Embedding 整合成一个；
3. 然后将整合后的这个 Embedding 作为 X 句在自己任务的那个网络结构中对应单词的输入，以此作为补充的新特征给下游任务使用。
4. 对于上图所示下游任务 QA 中的回答句子 Y 来说也是如此处理。

**因为 ELMo 给下游提供的是每个单词的特征形式，所以这一类预训练的方法被称为 “Feature-based Pre-Training”。**

至于为何这么做能够达到区分多义词的效果，原因在于在训练好 ELMo 后，**在特征提取的时候，每个单词在两层 LSTM 上都会有对应的节点，这两个节点会编码单词的一些句法特征和语义特征，并且它们的 Embedding 编码是动态改变的**，会受到上下文单词的影响，周围单词的上下文不同应该会强化某种语义，弱化其它语义，进而就解决了多义词的问题。





## 5.RNN和LSTM

RNN（Recurrent Neural Network） 和 LSTM（Long Short-Term Memory）

### RNN

* 传统的神经网络无法获取时序信息，然而**时序信息在自然语言处理任务中非常重要**
* RNN 的基本单元结构如下图所示![image-20240925152242019](https://img-blog.csdnimg.cn/direct/c2ce040182ce479d9047347cc4fc23f4.png)

上图左边部分称作 RNN 的一个 timestep，在这个 timestep 中可以看到，在 $t$ 时刻，输入变量 $x_t$，通过 RNN 的一个基础模块 A，输出变量 $h_t$，而 $t$ 时刻的信息，将会传递到下一个时刻 $t+1$

如果把模块按照时序展开，则会如上图右边部分所示，**由此可以看到 RNN 为多个基础模块 A 的互连，每一个模块都会把当前信息传递给下一个模块**。

RNN 解决了时序依赖问题，但这里的时序一般指的是**短距离**的，首先我们先介绍下短距离依赖和长距离依赖的区别：

- 短距离依赖：对于这个填空题 “我想看一场篮球____”，我们很容易就判断出 “篮球” 后面跟的是 “比赛”，这种短距离依赖问题非常适合 RNN。
- 长距离依赖：对于这个填空题 “我出生在中国的瓷都景德镇，小学和中学离家都很近，……，我的母语是____”，对于短距离依赖，“我的母语是” 后面可以紧跟着 “汉语”、“英语”、“法语”，但是如果我们想精确答案，则必须回到上文中很长距离之前的表述 “我出生在中国的瓷都景德镇”，进而判断答案为 “汉语”，而 RNN 是很难学习到这些信息的。

#### [RNN梯度消失问题](https://blog.csdn.net/zhaojc1995/article/details/114649486)

* 为什么**RNN不适合长距离依赖问题**

<img src="https://img-blog.csdnimg.cn/direct/14b5df8af70c486d8d0851038a4fee26.png" alt="image-20240925153100124" style="zoom:70%;" />

如上图所示，为RNN模型结构，前向传播过程包括：

* 隐藏状态: $h^{t}=\sigma({z^{(t)}}) = \sigma(Ux^{t}+Wh^{(t-1)}+b)$,此处激活函数一般为 tanh

* 模型输出：$o^{(t)}=Vh^{(t)}+c$

* 预测输出：$\hat{y}^2=\sigma(o^{(t)})$,此处激活函数一般为softmax。

* 模型损失：$L =\sum_{t=1}^NL^{(t)}$

  RNN 所有的 timestep 共享一套参数 U,V,W，在 RNN 反向传播过程中，需要计算 U,V,W等参数的梯度，以 W 的梯度表达式为例（假设 RNN 模型的损失函数为 L）：

![image-20240925154324891](https://img-blog.csdnimg.cn/direct/6774cdb7aea643be947368f7f3f9d350.png)

需要注意的是，RNN和DNN梯度消失和梯度爆炸含义并不相同

RNN中权重在各时间步内共享，最终的梯度是各个时间步的梯度和，梯度和会越来越大。因此，RNN中总的梯度是不会消失的，即使梯度越传越弱，也只是远距离的梯度消失。 从公式（9）中的$(\prod_{k=t+1}^Ttanh^\prime{(z^{(k)}W)}$可以看到，**RNN所谓梯度消失的真正含义是，梯度被近距离（$t+1趋向于T$）梯度主导，远距离（$t+1远离T$）梯度很小，导致模型难以学到远距离的信息。**

### LSTM 

为了解决 RNN 缺乏的序列长距离依赖问题，LSTM 被提了出来

![image-20240925155356045](https://img-blog.csdnimg.cn/direct/ee36d17132d54828b2bdd229f3f1fda1.png)

如上图所示，为 LSTM 的 RNN 门控结构（LSTM 的 timestep），LSTM 前向传播过程包括

* 遗忘门：

  **决定了丢弃哪些信息，**遗忘门接收$t−1$时刻的状态$h_{t−1}$，以及当前的输入$x_t$，经过 Sigmoid 函数后输出一个 0 到 1 之间的值$f_t$

  - 输出：$i_t=σ(W^ih_{t−1}+U_ix_t+b_i), \widehat{C}_t=tanhW_ah_{t−1}+U_ax_t+b_a$

* 输入门：**决定了哪些新信息被保留**，并更新cell状态，输入门的取值由 $h_{t−1}$ 和 ${x_t}$决定，通过 Sigmoid 函数得到一个 0 到 1 之间的值 $i_t$，而 $tanh$ 函数则创造了一个当前cell状态的候选 $a_t$

  * 输出：$i_t=\sigma(W_ih_{t-1}+U_ix_t+b_i) , \tilde{C}_t=tanhW_ah_{t-1}+U_ax_t+b_a$

* cell状态：旧cell状态 $C_{t−1}$ 被更新到新的cell状态 $C_t$ 上

  * 输出：$C_t=C_{t-1}\odot f_t+i_t\odot\tilde{C_t}$

* 输出门：决定了最后输出的信息，输出门取值由$h_{t−1}$ 和$x_{t}$决定，通过 Sigmoid 函数得到一个 0 到 1 之间的值 $o_t$，最后通过 $tanh$ 函数决定最后输出的信息

  * 输出$o_t=\sigma(W_oh_{t-1}+U_ox_t+b_o) , h_t=o_t\odot tanhC_t$

* 预测输出：$\hat{y}_t=\sigma(Vh_t+c)$

#### [LSTM解决RNN梯度消失问题](https://blog.csdn.net/zhaojc1995/article/details/114649486)

 1、cell state传播函数中的“加法”结构确实起了一定作用，它使得导数有可能大于1；
2、LSTM中逻辑门的参数可以一定程度控制不同时间步梯度消失的程度。

最后，LSTM依然不能完全解决梯度消失这个问题，有文献表示序列长度一般到了三百多仍然会出现梯度消失现象。如果想彻底规避这个问题，还是transformer好用

## 6.Attention

### 本质思想

虽然 LSTM 解决了序列长距离依赖问题，但是单词超过 200 的时候就会失效。**而 Attention 机制可以更加好的解决序列长距离依赖问题，并且具有并行计算能力**

首先我们得明确一个点，注意力模型从大量信息 Values 中筛选出少量重要信息，**这些重要信息一定是相对于另外一个信息 Query 而言是重要的**。也就是说，我们要搭建一个注意力模型，我们必须得要有一个 Query 和一个 Values，然后通过 Query 这个信息从 Values 中筛选出重要信息。简单点说，**就是计算 Query 和 Values 中每个信息的相关程度。**

![image-20240926141606010](https://img-blog.csdnimg.cn/direct/79464bde78ad46baabef1ab09061cbbc.png)

通过上图，Attention 通常可以进行如下描述，表示为将 Query(Q) 和 key-value pairs（**把 Values 拆分成了键值对的形式**） 映射到输出上，其中 query、每个 key、每个 value 都是向量，输出是 V 中所有 values 的加权，其中权重是由 Query 和每个 key 计算出来的，计算方法分为三步：

* 第一步：计算比较 Q 和 K 的相似度，用 f 来表示：$f(Q,K_i)\quad i=1,2,\cdots,m$,一般第一步计算方法包括四种
  * 点乘(**Transformer使用**):$f(Q,K_i)=Q^TK_i$
  * 权重：$f(Q,K_i)=Q^TWK_i$
  * 拼接权重：$: f(Q,K_i)=W[Q^T;K_i]$
  * 感知器：$f(Q,K_i)=V^T\tanh(WQ+UK_i)$
* 第二步：将得到的相似度进行 softmax 操作，进行归一化：$$\alpha_{i}=softmax(\frac{f(Q,K_{i})}{\sqrt{d}_{k}})$$
  * 为什么除以$\sqrt{d}_{k}$: 假设 Q , K 里的元素的均值为0，方差为 1，那么 $A^T=Q^TK$中元素的均值为 0，方差为 d。当 d 变得很大时， A 中的元素的方差也会变得很大，如果 A 中的元素方差很大(分布的方差大，分布集中在绝对值大的区域)，**在数量级较大时， softmax 将几乎全部的概率分布都分配给了最大值对应的标签**，由于某一维度的数量级较大，进而会导致 softmax 未来求梯度时会消失。
  * 总结一下就是 $softmax⁡(A)$ 的分布会和d有关。因此 AA中每一个元素乘上 $\frac{1}{\sqrt{d}_{k}}$ 后，**方差又变为 1，**并且 A 的数量级也将会变小。
* 第三步：针对计算出来的权重 αi，对 V 中的所有 values 进行加权求和计算，得到 Attention 向量:$Attention=\sum_{i=1}^m\alpha_iV_i$

### self-attention

![image-20240926144716980](https://img-blog.csdnimg.cn/direct/365497885a4f4919aaa45ed880a51b68.png)

首先可以看到 Self Attention 有三个输入 Q、K、V：**对于 Self Attention，Q、K、V 来自句子 X 的 词向量 x 的线性转化，即对于词向量 x，给定三个可学习的矩阵参数 $W_Q,W_k,W_v$，x 分别右乘上述矩阵得到 Q、K、V**。

**计算流程**

* 第一步，Q、K、V 的获取![image-20240926144847487](https://img-blog.csdnimg.cn/direct/516cbfebb23d42c69ac4584dc1707987.png)

上图操作：两个单词 Thinking 和 Machines。通过线性变换，即$x_1 和 x_2$两个向量分别与$W_Q,W_k,W_v$ 三个矩阵点乘得到 $q_1,q_2,k_1,k_2,v_1,v_2 共 6 个向量。矩阵 Q 则是向量$ $q_1,q_2$ 的拼接，K、V 同理。

* 第二步，MatMul![image-20240926145106372](https://img-blog.csdnimg.cn/direct/bc227fa5da5b46709d2715dd405cd409.png)

上图操作：向量 $q_1,k_1$做点乘得到得分 112， $q_1,k_2$ 做点乘得到得分96。注意：**这里是通过 $q_1$这个信息找到 $x_1,x_2$ 中的重要信息。**

* 第三步和第四步，Scale + Softmax

![image-20240926145325955](https://img-blog.csdnimg.cn/direct/e6e5f5b56c994593926e2b559325baa2.png)

对该得分进行规范，除以 $\sqrt{d_k}=8$

* 第五步，MatMul

![image-20240926145542828](https://img-blog.csdnimg.cn/direct/489f71f3ec2c40b8945d1ae89dbdd633.png)

用得分比例 [0.88，0.12] 乘以 $[v_1,v_2]$ 值得到一个加权后的值，将这些值加起来得到 $z_1$

上述所说就是 Self Attention 模型所做的事，仔细感受一下，用$q_1$、$K=[k_1,k_2]$去计算一个 Thinking 相对于 Thinking 和 Machine 的权重，再用权重乘以 Thinking 和 Machine 的$V=[v_1,v_2]$ 得到加权后的 Thinking 和 Machine 的$V=[v_1,v_2]$,最后求和得到针对各单词的输出$z_{1}$
同理可以计算出 Machine 相对于 Thinking 和 Machine 的加权输出$z_2$,拼接$z_1$和$z_2$即可得到 Attention 值$Z=[z_1,z_2]$,这就是 Self
Attention 的矩阵计算，如下所示。
之前的例子是单个向量的运算例子。这张图展示的是矩阵运算的例子，输入是一个[2x4]的矩阵(句子中每个单词的词向量的拼接),每
个运算是[4x3]的矩阵，求得Q、K、V。

![image-20240926145931157](https://img-blog.csdnimg.cn/direct/eee5f81a1cde499494b5c588a5820ba7.png)

Q对K转制做点乘，除以$\sqrt{d}_k$,做一个 softmax 得到合为 1 的比例，对V做点乘得到输出 Z。那么这个 Z 就是一个考虑过Thinking 周围单词Machine 的输出。

![image-20240926150145483](https://img-blog.csdnimg.cn/direct/f4fffeb6e49a49ccadae84154f05c3d3.png)



**Self Attention 和 RNN、LSTM 的区别**

- RNN、LSTM：如果是 RNN 或者 LSTM，需要依次序列计算，对于远距离的相互依赖的特征，**要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小**。
- Self Attention：
  - 引入 Self Attention 后会更容易捕获句子中长距离的相互依赖的特征，**因为 Self Attention 在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征**；
  - 除此之外，Self Attention 对于**一句话中的每个单词都可以单独的进行 Attention 值的计算**，也就是说 Self Attention 对计算的并行性也有直接帮助作用，而对于必须得依次序列计算的 RNN 而言，是无法做到并行计算的。

#### Masked Self Attention 模型

![image-20240926150727812](https://img-blog.csdnimg.cn/direct/7b91122adc5e465aaee55afc24a9b737.png)

我们已经通过 scale 之前的步骤得到了一个 attention map，**而 mask 就是沿着对角线把灰色的区域用0覆盖掉，不给模型看到未来的信息**，如下图所示![image-20240926150747585](https://img-blog.csdnimg.cn/direct/9fb4f7d68ec24fc4ba50ed23044bc593.png)

在做完 softmax 之后，横轴结果合为 1

#### Multi-head Self Attention 模型

$\text{Multi-Head Attention 就是把 Self Attention 得到的注意力值}Z\text{切分成}\text{n个}Z_1,Z_2,\cdots,Z_n\\\text{,然后通过全连接层获得新的 }Z^{\prime}$

<img src="https://img-blog.csdnimg.cn/direct/3eda42531c8c48f995195263c07c4655.png" alt="image-20240926151121502" style="zoom:50%;" />

我们对 $Z$ 进行 8 等份的切分得到 8 个 $Z_i$ 矩阵

<img src="https://img-blog.csdnimg.cn/direct/7123375216ab466183021b21fbf8b98e.png" alt="image-20240926151206524" style="zoom:67%;" />

为了使得输出与输入结构相同，拼接矩阵  $Z_i$  后乘以一个线性  $W_o$ 得到最终的 $Z$ 

<img src="https://img-blog.csdnimg.cn/direct/f7d3b1813e1746709aff99ca2b79cc94.png" alt="image-20240926151303986" style="zoom:50%;" />



整个流程：<img src="https://img-blog.csdnimg.cn/direct/088bb4ea3ee14572871cff4b21a1ea17.png" alt="image-20240926151344964" style="zoom:67%;" />

**多头相当于把原始信息 Source 放入了多个子空间中，也就是捕捉了多个信息，对于使用 multi-head（多头） attention 的简单回答就是，多头保证了 attention 可以注意到不同子空间的信息，捕捉到更加丰富的特征信息**

## 7.zero-shot few-shot Learning

[概括](https://blog.csdn.net/zcyzcyjava/article/details/127006287)

[zero-shot learning](https://www.ibm.com/cn-zh/topics/zero-shot-learning)

[few-shot learning](https://www.ibm.com/cn-zh/topics/few-shot-learning)



## 8.N-gram

[一文读懂](https://blog.csdn.net/songbinxu/article/details/80209197)

## 9.分词

[一文读懂](https://juejin.cn/post/7088322473640329230)

## 10.Genetic Algorithm算法

**overall：**

是一种用于解决[最优化](https://zh.wikipedia.org/wiki/最佳化)的搜索[算法](https://zh.wikipedia.org/wiki/算法)，是[进化算法](https://zh.wikipedia.org/wiki/进化算法)的一种。进化算法最初是借鉴了[进化生物学](https://zh.wikipedia.org/wiki/进化生物学)中的一些现象而发展起来的，这些现象包括[遗传](https://zh.wikipedia.org/wiki/遗传)、[突变](https://zh.wikipedia.org/wiki/突变)、[自然选择](https://zh.wikipedia.org/wiki/自然选择)以及[杂交](https://zh.wikipedia.org/wiki/杂交)等等

对于一个最优化问题，一定数量的[候选解](https://zh.wikipedia.org/w/index.php?title=候选解&action=edit&redlink=1)（称为个体）可抽象表示为[染色体](https://zh.wikipedia.org/wiki/染色體_(遺傳演算法))，使[种群](https://zh.wikipedia.org/wiki/种群)向更好的解进化。传统上，解用[二进制](https://zh.wikipedia.org/wiki/二进制)表示（即0和1的串），但也可以用其他表示方法。进化从完全[随机](https://zh.wikipedia.org/wiki/随机)个体的种群开始，之后一代一代发生。在每一代中评价整个种群的[适应度](https://zh.wikipedia.org/wiki/适应度)，从当前种群中随机地选择多个个体（基于它们的适应度），通过自然选择和突变产生新的生命种群，该种群在算法的下一次迭代中成为当前种群

**算法原理：**

首先是编码过程，染色体一般表示为字符串或数字符串。算法[随机](https://zh.wikipedia.org/wiki/随机函数)生成一定数量的个体，有时候操作者也可以干预这个随机产生过程，以提高初始种群的质量，在每一代中，都会评价每一个体，并通过计算[适应度函数](https://zh.wikipedia.org/w/index.php?title=适应度函数&action=edit&redlink=1)得到[适应度](https://zh.wikipedia.org/wiki/适应度)数值。按照适应度[排序](https://zh.wikipedia.org/wiki/排序)种群个体，适应度高的在前面。这里的“高”是相对于初始的种群的低适应度而言。

下一步，产生下一代个体并组成种群。这个过程是通过选择和[繁殖](https://zh.wikipedia.org/wiki/繁殖)完成，其中繁殖包括交配（crossover，在算法研究领域中我们称之为交叉操作）和突变（mutation）。选择则是根据新个体的适应度进行，但同时不意味着完全以适应度高低为导向，因为单纯选择适应度高的个体将可能导致算法快速收敛到局部最优解而非全局最优解，我们称之为早熟。作为折中，遗传算法依据原则：适应度越高，被选择的机会越高，而适应度低的，被选择的机会就低。初始的数据可以通过这样的选择过程组成一个相对优化的群体。之后，被选择的个体进入交配过程。一般的遗传算法都有一个交配概率（又称为交叉概率），范围一般是0.6~1，这个交配概率反映两个被选中的个体进行交配的[概率](https://zh.wikipedia.org/wiki/概率)。例如，交配概率为0.8，则80%的“夫妻”会生育后代。每两个个体通过交配产生两个新个体，代替原来的“老”个体，而不交配的个体则保持不变。交配父母的染色体相互交换，从而产生两个新的染色体，第一个个体前半段是父亲的染色体，后半段是母亲的，第二个个体则正好相反。不过这里的半段并不是真正的一半，这个位置叫做交配点，也是随机产生的，可以是染色体的任意位置。再下一步是[突变](https://zh.wikipedia.org/wiki/突變)，通过突变产生新的“子”个体。一般遗传算法都有一个固定的[突变常数](https://zh.wikipedia.org/w/index.php?title=突变常数&action=edit&redlink=1)（又称为变异概率），通常是0.1或者更小，这代表变异发生的概率。根据这个概率，新个体的染色体随机的突变，通常就是改变染色体的一个字节（0变到1，或者1变到0）

经过这一系列的过程（选择、交配和突变），产生的新一代个体不同于初始的一代，并一代一代向增加整体适应度的方向发展，因为总是更常选择最好的个体产生下一代，而适应度低的个体逐渐被淘汰掉。这样的过程不断的重复：评价每个个体，计算适应度，两两交配，然后突变，产生第三代。周而复始，直到终止条件满足为止。一般终止条件有以下几种：

- 进化次数限制；
- 计算耗费的资源限制（例如计算时间、计算占用的内存等）；
- 一个个体已经满足最优值的条件，即最优值已经找到；
- 适应度已经达到饱和，继续进化不会产生适应度更好的个体；
- 人为干预；
- 以及以上两种或更多种的组合

以上过程可以总结为

```
        +---------------------+
        |  初始化种群        |
        |  随机生成个体或    |
        |  干预生成较优个体  |
        +---------------------+
                 |
                 v
        +---------------------+
        |  计算适应度        |
        |  评估每个个体的适应度 |
        +---------------------+
                 |
                 v
        +---------------------+
        |  选择操作          |
        |  根据适应度选出个体 |
        +---------------------+
                 |
                 v
        +---------------------+
        |  交配操作          |
        |  交叉产生新个体    |
        +---------------------+
                 |
                 v
        +---------------------+
        |  突变操作          |
        |  根据突变概率改变基因 |
        +---------------------+
                 |
                 v
        +---------------------+
        |  生成新一代种群    |
        |  替换低适应度个体  |
        +---------------------+
                 |
                 v
        +---------------------+
        |  重复评估、选择和交配 |
        |  直到满足终止条件  |
        +---------------------+
                 |
                 v
        +---------------------+
        |  终止条件          |
        |  满足进化次数或找到最优解 |
        +---------------------+

```

**算法：**

> 步骤

- 选择初始生命种群
- 循环
  - 评价种群中的个体适应度
  - 以比例原则（分数高的挑中几率也较高）选择产生下一个种群（[轮盘法](https://zh.wikipedia.org/w/index.php?title=輪盤法&action=edit&redlink=1)（roulette wheel selection）、[竞争法](https://zh.wikipedia.org/wiki/競爭法)（tournament selection）及[等级轮盘法](https://zh.wikipedia.org/w/index.php?title=等級輪盤法&action=edit&redlink=1)（Rank Based Wheel Selection））。不仅仅挑分数最高的的原因是这么做可能收敛到局部的最佳点，而非整体的。
  - 改变该种群（交叉和变异）
- 直到停止循环的条件满足.

> 参数

- 种群规模（P,population size）：即种群中染色体个体的数目。
- 字符串长度（l, string length）：个体中染色体的长度。
- 交配概率（pc, probability of performing crossover）：控制着交配算子的使用频率。交配操作可以加快收敛，使解达到最有希望的最佳解区域，因此一般取较大的交配概率，但交配概率太高也可能导致过早收敛，则称为早熟。
- 突变概率（pm, probability of mutation）：控制着突变算子的使用频率。
- 中止条件（termination criteria）

[简单的onemax示例](https://blog.csdn.net/LOVEmy134611/article/details/111639624)







[click here](https://www.cnblogs.com/nickchen121/p/16470569.html#tid-EEyxQf)