---
title: bert
abbrlink: 1bb262dc
date: 2024-09-25 21:45:41
updated: 2024-09-25 10:14:12
categories: paper reading
mathjax: true
toc: true
---
<meta name="referrer" content="no-referrer"/>


# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[Paper](https://arxiv.org/pdf/1810.04805.pdf) | [Talk](https://www.bilibili.com/video/BV1PL411M7eQ)

## Part1. 标题&作者
Pre-training: 用来做预训练的

Bidirectional Transformers: 双向的transformer

Language Understanding: transformer主要用于机器翻译，bert将其扩展为语言理解任务，更加广泛

**BERT**：它是一个深的双向的transformer，是用来做预训练的，面向的是一般的语言理解任务

## Part2. 摘要
**BERT名字来由**：**B**idirectional **E**ncoder **R**epresentations from **T**ransformers
> ELMo, BERT, ERNIE均是美国著名儿童教育节目《芝麻街》中的人物

**BERT与GPT和ELMo的区别**：
- BERT: 使用未标注数据，联合左右的上下文信息，训练得到深的双向的表示；得益于此设计，针对不用的下游语言（问答、推理等）任务，只需要对预训练好的BERT模型另加一个额外的输出层即可
- GPT: 是一个单向过程，使用左边的上下文信息去预测未来，而BERT同时使用左侧和右侧信息（Bidirectional双向的由来）
- ELMo: 基于RNN架构，BERT是基于transformer；ELMo在用到一些下游任务上时通常需要对架构做一些调整，而BERT只需要改最上层即可

## Part3. 导言
**NLP任务可分为两类**:
- sentence-level tasks: 句子层面的任务，主要用来建模句子之间的关系，比如句子情绪识别、两个句子之间的关系等
- token-level tasks: 词元层面的任务，比如实体命名识别（NER）等

**在使用预训练模型做特征表示时，一般用两类策略**：
- feature-based:对每一个下游任务，构造一个与该任务相关的神经网络（比如RNN结构），而预训练好的表示（可理解为预训练模型的输出）作为额外的特征和已有输入一起送给模型，代表作ELMo
- fine-tuning:在预训练模型用于下游任务时，只需要针对任务修改头部结构即可，模型预训练好的参数会在下游任务的数据上再进行微调一下（即所有的权重都会进行微调，对比feature-based来看由于输入的是特征相当于预训练模型的参数是固定的），代表作GPT

**当前预训练模型技术的局限性**：

- 标准的语言模型是一个单向的任务，导致选架构时有一些局限性，比如GPT使用的是一个从左到右的架构；但一些任务无论从左到右或者从右到左结构应该是一样的，比如句子情绪识别

**BERT是如何突破以上局限性的？**
- “masked language model”(MLM)：带掩码的语言模型，对句子随机掩盖部分词汇，然后预测这些词（完形填空），允许同时看左右的信息（与从左到右不同）
- “next sentence prediction”: 给定两个句子，判断这两个句子在原文中是否相邻，希望可以学习到句子层面的一些信息
> 我理解以上两种方式分别想针对token-level与sentence-level任务

## Part4. 结论
- BERT将之前基于单向结构的预训练模型拓展到双向结构，使得同样的一个预训练模型能够处理不同的NLP任务
- BERT是ELMo使用双向信息（但基于比较旧的RNN架构）与GPT使用新架构transformer（但只能处理单项信息）两者的结合

## Part5. 相关工作
- Unsupervised Feature-based Approaches: 代表作ELMo
- Unsupervised Fine-tuning Approaches: 代表作GPT
- Transfer Learning from Supervised Data: 在有标注的数据上，针对下游任务进行迁移学习

## Part6. BERT模型
**BERT中的两个步骤**：pre-training和fine-tuning

<img src="https://user-images.githubusercontent.com/22740819/145942921-7ed8cde2-d193-45bd-b169-dfed380de035.png" width=600>

- 首先pre-training阶段使用无标注数据进行训练得到预训练模型，然后fine-tuning阶段使用标注数据对模型进行微调
- fine-tuning时每一个下游任务都会有一个自己特定的bert模型，由预训练好的模型进行参数初始化（输出层除外），然后在自己任务的标注数据上进行微调，期间所有参数都会参与训练

### 模型架构
BERT模型是一个多层的双向的transformer的编码器

**模型的三个可调整参数**：
- L：transformer block的个数
- H：隐藏层大小
- A：self-attention head的个数

> BERT_base（L=12, H=768, A=12, Total Parameters=110M），BERT_large（L=24, H=1024, A=16, Total Parameters=340M）

**可学习参数量的计算**

BERT模型里的可学习参数主要来自两部分：嵌入层与Transformer Block

<img src="https://user-images.githubusercontent.com/22740819/145948712-57dd1b13-596a-464f-a216-c2a39fb12ad6.png" width=400>

***嵌入层***：实际是一个矩阵，输入为字典的大小30k，输出为隐层单元的个数H；
> 参数量=30k（字典大小）* H（hidden size）

***Transformer块***：包括multi head self-attention与MLP两部分中的参数，self-attention本身不包含参数，但是multi head self-attention中会把所有进入的Q、K、V分别做一次投影，每次投影的维度为64（`A * 64=H`）；MLP包含两个全连接层（第一个全连接层输入是`H`，输出是`4 * H`，参数量`H^2 * 4`；第二个全连接层输入是`4 * H`，输出是`H`，参数量`H^2 * 4`）
> Multi head self-attention参数量=`H^2 * 4`，MLP参数量=`H^2 * 4 + H^2 * 4 = H^2 * 8`，整个transformer block的参数量为`H^2 * 4 + H^2 * 8 = H^2 * 12`，L 个 blocks的参数量=`L * H^2 * 12`

BERT总参数量=`30k * H`（嵌入层）+ `L * H^2 * 12`（L个transformer块），以BERT_base为例`H=768`，`L=12`，总参数量=1.1亿（即`110M`）

（该部分建议看视频更直观，视频于23:22处）

**BERT的输入和输出**

为了处理不同的下游任务，BERT的输入即可以是一个句子（一段连续的文字，不一定是真正的语义上的一段句子），也可以是一个句子对；无论是一个句子还是句子对，它们最终都是构成一个序列（sequence）作为输入；
> BERT输入与Transformer输入的区别：transformer的输入是一个序列对，编码器和解码器分别会输入一个序列；而BERT只有一个编码器，为了使BERT能处理两个句子的情况，需要把两个句子并成一个序列。

**BERT输入的序列是如何构成的**

***切词***

假设按照空格切词，每个词作为一个token，如果数据量比较大，会导致切出的词典特别大，可能是百万级别，由之前计算模型参数的方法可知，若是百万级别就会导致模型的整个可学习参数都在嵌入层上。

WordPiece，如果一个词出现的频率比较低，则应该把它切开，然后只保留一个词频率较高的子序列。这样的话，可以把一个相对来说比较长的文本，切成很多片段，而且这些片段是经常出现的，进而可以用一个相对来说比较小的词典（30k）就可以表示一个比较大的文本了。

***序列构成***

BERT的输入序列，第一个词永远是[CLS]（classification），其输出代表整个序列的信息

> [CLS]：BERT希望它代表的是整个序列的信息，因为transformer中的self-attention机制使得输入的每一个token都会去看其它所有的token信息，就算[CLS]放在序列的第一个位置，它也是有办法看到之后所有的词

将两个句子合成一个序列之后，为了区分开两个句子，所采用了两种方式：
- 在每一个句子后面放一个特殊词[SEP]（separate）；
- 添加一个可学习的嵌入层，来表示当前句子到底是第一个句子还是第二个句子。

> [SEP]：用于区分两个句子的连接符

最终所构成的序列：`[CLS][Token1]...[TokenN][SEP][Token1']...[TokenN']`

***输入-输出向量化表示***

<img src="https://user-images.githubusercontent.com/22740819/145942921-7ed8cde2-d193-45bd-b169-dfed380de035.png" width=600>

在将句子切词、序列化之后，送入transformer之前，是需要对输入序列（每个token）进行向量化表示的（上图中黄色部分）；每个token被embedding之后送入transformer，然后输出该token新的embedding表示（上图中绿色部分）。所以，BERT是一个序列到序列的模型。针对下游任务，添加额外的输出层，将由bert得到的token embedding映射为想要的结果。


<img src="https://user-images.githubusercontent.com/22740819/145993078-ef5cc313-f930-4e0f-8f20-d716499e28b3.png" width=600>

对于每个词元token进入BERT的向量表示（token embedding），是由token、segment以及position embedding三者相加获得。
- token embedding: token本身的embedding
- segment embedding: 该token是属于第一个句子还是第二个句子的embedding
- position embedding: 该token在句子中位置的embedding

### BERT Pre-training部分

预训练阶段两个关键部分：目标任务和预训练的数据

**目标任务1: Masked LM**

Masked LM实际是一个**完形填空**任务。对于一个输入BERT的词元序列，如果一个token是有WordPiece生成的，那么它有15%的的概率被替换成一个掩码，但是对于特殊的token不做替换（[ CLS ]和[SEP]）。假如输入序列是1000个token，那么该任务需要预测其中的150个被mask掉的词。 

Masked LM存在的问题：在做掩码时，会将词元替换成特殊符号[MASK]，所以在pre-training阶段会看到输入中有15%的[MASK]，但是在fine-tuning阶段是没有[MASK]这个东西的（因为微调时不用Masked LM这个目标任务）
> 解决方法：对于被选中的15%去做掩码的token，有80%的概率真的将其替换成[MASK]，10%的概率将其替换成一个随机的token，还有10%的概率该token保持不变，但是让它用于预测使用。

**目标任务2: Next Sentence Prediction(NSP)**

一个给定的序列包含两个句子a和b，其中有50%的概率在原文之中句子b确实是在句子a之后，还有50%的概率句子b是从另外一个地方随机选取出来的一个句子（其实就是50%的样本为正例，50%的样本为负例），该任务就是预测句子b是否为句子a的下一句。

该预训练任务对QA、推理等下游任务有明显的增益。

**Pre-training data**

- BooksCorpus：具有8亿个词
- Englishi Wikipedia：25亿个词

预训练时应该使用文本级别（document-level）的预料，即送进去的时一篇一篇的文章，而不是一些随机打乱的句子。因为Transformer本身可以处理比较长的序列，输入整体的文本效果会更好一些。

### BERT Fine-tuning部分

**BERT与一些基于encoder-decoder架构的模型（如transformer）的区别？**

- BERT预训练时的输入是整个文章或者一些句子对，由于self-attention机制导致可以在两端之间相互看到，但是基于encoder-decoder的结构里，一般encoder是看不到decoder的东西的，这里BERT会更好一些（不太理解），所付出的代价是BERT不能像Transformer那样来做机器翻译任务了。

**下游任务**

针对下游任务，只需要设计输入和输出，模型架构不需要变。即只需考虑你的输入如何改造成BERT所需要的句子对：

- 如果你的下游任务也有两个句子作为输入，那就是句子A和B
- 如果你的下游任务也有一个句子作为输入（比如句子分类），那么相当于句子B是省略的

## Part7. 实验

**任务1:GLUE (General Language Understanding Evaluation)**

这是一个句子层面的分类任务，[CLS]的BERT输出加一个输出映射就可以得到最终的输出。

**任务2:SQuAD (Stanford Question Answering Dataset)**

这是斯坦福的一个QA数据集，QA任务就是给定一段话，然后问你一个问题，需要你把答案找出来。答案已经在给定的那段话中，只需要把答案对应的那个片段找出来就可以（片段的开始和结尾），实际操作时就是对每一个token，来判断一下它是不是答案的开头或者结尾，具体的就是对每个token学习得到两个向量S和E，分别代表其作为答案的开始和结尾的概率。

**任务3: SQuAD 2.0**

同上

**任务4:SWAG**

这是一个用来判断两个句子之间关系的任务。

## Part8. 评论

略。

## 八股相关问题

1. BERT分为哪两种任务，各自的作用是什么？

Masked Language Model（MLM，遮盖语言模型）：这种任务的目的是预测句子中部分单词的原始形式。在训练过程中，BERT模型会随机选择一些单词并用“【MASK】”标记替换它们。模型的任务是预测被替换的单词的原始形式。这种方法可以使模型在理解句子语义的同时学习到词语之间的关系

Next Sentence Prediction（NSP，下一句预测）：这种任务的目的是预测一个句子是否是另一个句子的下一句。在训练过程中，BERT模型会从两个句子中选择一个随机的句子对，并根据是否是下一句来训练模型。这种方法可以使模型更好地理解上下文之间的关系。

2. 在计算MLM预训练任务的损失函数的时候，参与计算的Tokens有哪些？是全部的15%的词汇还是15%词汇中真正被Mask的那些tokens？

在训练过程中，BERT模型会随机选择一些tokens并用“【MASK】”标记替换它们。这些被标记的tokens只占所有tokens的一小部分，通常是15%。然后，模型的任务是预测被替换的tokens的原始形式。在计算损失函数时，只有真正被替换成了“【MASK】”标记的tokens会被用来计算损失值，而没有被标记的tokens则不会参与损失函数的计算

3. 在实现损失函数的时候，怎么确保没有被 Mask 的函数不参与到损失计算中去？

在实现损失函数时，需要使用一个掩码（mask）向量来指示哪些tokens是被Mask的，哪些tokens是没有被Mask的。具体来说，掩码向量中被Mask的tokens的位置为1，没有被Mask的tokens的位置为0。在计算损失函数时，可以将掩码向量与预测的tokens和实际的tokens相乘，这样就可以将没有被Mask的tokens的损失值置为0，只计算被Mask的tokens的损失值。以PyTorch为例，可以使用以下代码来实现掩码的功能：
~~~python 
loss_mask = torch.tensor(mask, dtype=torch.float32) # mask是掩码向量
predictions = model(tokens) # tokens是输入的tokens
loss = loss_function(predictions, labels)
masked_loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) 
~~~

4. BERT的三个Embedding为什么直接相加？

这样做的原因是因为BERT模型需要同时考虑输入的token、位置和段落信息。这三个嵌入分别对应了输入token在句子中的位置和所处的段落，以及输入token本身的信息。因此，在将它们相加之后，BERT模型可以同时获得这些信息

5. BERT的优缺点分别是什么？

BERT模型的优点包括：
强大的语言表示能力：BERT模型使用了双向Transformer结构，能够学习到更丰富的语言表示，可以应对各种自然语言处理任务。
预训练模型的通用性：BERT模型是一种预训练模型，能够在大规模无标注数据上进行预训练，然后在有标注数据上进行微调，可以适用于各种任务和语言。

BERT模型也存在一些缺点：
模型过于复杂：BERT模型拥有数亿个参数，需要在GPU或TPU等硬件平台上进行训练和推理，对计算资源的要求较高。
学习时间较长：由于BERT模型需要进行预训练，因此其训练时间相对较长，需要耗费大量的计算资源。
预训练数据集需要大规模无标注数据：BERT模型的预训练需要大规模无标注数据集，这对于资源有限的组织和个人来说可能是一个限制因素

6. 你知道有哪些针对BERT的缺点做优化的模型？

SpanBERT: SpanBERT是一种针对自然语言推理（NLI）任务的BERT模型改进，它通过对输入序列中的部分单词进行特殊标记，来使模型能够更好地理解上下文中的语义关系;通过剪枝和蒸馏技术来减少模型的大小和计算量，从而提高模型的训练速度和推理速度。

7. BERT怎么用在生成模型中

Seq2Seq模型：使用BERT模型作为编码器，将输入序列转换为上下文向量，然后将上下文向量输入到解码器中进行解码生成输出序列。该方法可以用于生成对话系统、机器翻译、文本摘要等任务。

GPT模型：使用BERT模型进行预训练，然后将预训练好的BERT模型作为初始参数，使用Transformer解码器进行微调，从而实现对文本生成任务的优化。GPT模型可以用于生成自然语言文本、文章摘要、问答系统等任务