---
title: BERT-ATTACK Adversarial Attack Against BERT Using BERT
categories: paper reading
abbrlink: 4158q210
date: 2024-10-16 21:15:51
tags:
mathjax: true
toc: true
---
<meta name="referrer" content="no-referrer"/>

BERT-ATTACK Adversarial Attack Against BERT Using BERT

## Part 1. Title & Source

**Title**: BERT-ATTACK: Adversarial Attack Against BERT Using BERT\
**Source**: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), November 16–20, 2020.

## Part 2. Abstract

该论文提出了一种名为**BERT-Attack**的对抗样本生成方法，利用BERT预训练语言模型生成对抗样本，从而误导下游任务中的BERT和其他深度神经网络模型。BERT-Attack通过找到句子中的易受攻击词并用BERT生成语义保留的替换词，从而生成流畅的对抗样本。实验表明，与现有方法相比，BERT-Attack在成功率和扰动百分比上均有优势，且计算成本低，适合大规模生成对抗样本。

## Part 3. Introduction

深度学习模型虽然取得了显著成功，但它们容易受到对抗样本的攻击。现有的文本对抗样本生成方法大多采用基于启发式的替换策略，然而由于文本的离散性，这些方法面临保留语义一致性和语言流畅性的挑战。

论文提出的BERT-Attack利用了BERT的强大语言建模能力，通过在给定输入的上下文中生成替换词，从而保证对抗样本的流畅性和语义一致性。具体过程包括：1）找出输入序列中对模型预测影响最大的词汇；2）利用BERT生成语义保留的替换词。该方法使用掩码语言模型作为扰动生成器，并找到使做出错误预测的风险最大化的扰动。与之前的攻击策略相比，BERT-Attack方法更有效率，也无需大量重复推理。

## Part 4. Conclusion

BERT-Attack成功地生成了高质量、语义保留且具有欺骗性的对抗样本，在多个自然语言处理任务中显著降低了模型的准确性。相比于之前的文本攻击方法，BERT-Attack在保证语义一致性的同时，取得了更高的攻击成功率和更低的查询次数。这表明利用预训练的语言模型生成对抗样本是一种高效且鲁棒的策略。此外，BERT-Attack还适用于攻击不同的模型，不仅限于BERT。

## Part 5. Related Work

CV领域由于图片的连续空间上应用梯度下降，可以很容易的找到对抗性样本，而文本对抗攻击由于语言的离散性而更具挑战性。现有的文本攻击方法大多基于字符或词的**替换策略**，通常依赖于**词嵌入**或**启发式规则**，这些方法在生成语义一致且流畅的对抗样本方面效果有限。此外，论文还讨论了BERT在自然语言处理任务中的应用，指出BERT的强大能力也为对抗攻击提出了更高的挑战。

## Part 6. Model\&Algorithm

BERT-Attack的核心思想是利用BERT模型生成对抗样本，其主要步骤包括：

**1.找到易受攻击的词汇**：

根据目标模型的logit输出，计算每个词对最终预测的贡献分数，选择最重要的词汇作为攻击目标。重要性得分定义如下：

$I_{w_i}=o_y(S)-o_y(S_{\setminus w_i}),\quad(1)$

> $S_{\setminus w_i}=[w_0,\cdots,w_{i-1},[\text{мАSK}],w_{i+1},\cdots]$ 是将 wi 替换为 \[MASK] 后的句子

词汇重要性排名计算过程：

输入句子S（已分词），正确标签Y，遍历句子中的每个词wi，计算每个词对输出logit的影响，重要性分数$I_{w_i}$ ​ 表示如果去掉这个词，模型输出的变化。最后按重要性分数对所有词进行排序，选出前ε%的重要词汇，构成词汇列表L

**2. 通过bert词汇替换**：

使用BERT作为掩码语言模型，生成语义保留的替换词。与以往的方法不同，**BERT-Attack只需一次前向传播就能生成扰动**，而无需反复使用语言模型打分。

替换算法如下：

输入原句子H（子词分词），词汇列表L，对于列表L中的每个词wj：

*   检查如果wj是完整的单词，则获取其候选词C。候选词是通过BERT预测的最可能的K个替代词，经过停用词和反义词过滤。
*   如果wj是子词，则通过概率排名和过滤获取候选词。

最后生成可能的对抗性样本，将原句中的wj替换为候选词ck，形成新句子 $S^{\prime}=[w_0,...,w_{j-1},c_k,...]$ 。检查新句子的模型预测是否与正确标签Y不一致，即 $\operatorname{argmax}(o_y(S^{\prime}))\neq Y$ 。如果是，则返回生成的对抗样本Sadv，否则，记录下较优的对抗样本Sadv，继续对下一个词进行处理。

前两个步骤如下：

![\<img alt="" data-attachment-key="YU42J95F" width="653" height="667" src="attachments/YU42J95F.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/23c910a9311748b6a607d9ecda61f02f.png)

**3.扰动生成策略**：对于选定的词，使用BERT生成前K个候选词，并在不打断语义一致性的前提下进行替换，直到成功误导模型。

该模型通过对BERT的动态上下文感知能力进行有效利用，生成的对抗样本既具备流畅度，又保留了原始输入的大部分语义信息。

## Part 7. Experiment

**数据集**

文本分类：

*   Yelp Review：情感分析任务，分类为积极和消极。
*   IMDB：文档级电影评论分类，同样用于情感分析。
*   AG’s News：包含四种新闻类型（世界、体育、商业和科学）的句子级分类。
*   FAKE：假新闻分类任务，判断新闻是否真实

自然语言推理（NLI）：

*   SNLI：给定一个前提和假设，预测假设是否与前提存在蕴涵、矛盾或中立关系。
*   MNLI：多领域文本的语言推理数据集，包含多种文体的文本，更加复杂

**自动评估指标**

1.  **success rate**：对抗攻击后模型的准确率，越低表明攻击越成功。
2.  **perturbed percentage**：被替换的词汇数量占总词汇数量的比例，越低表明语义保持越好。
3.  **queries number**：进行对抗攻击时对目标模型的查询次数，越少越好。
4.  **semantic consistency**：使用通用句子编码器（Universal Sentence Encoder）测量原始样本与生成对抗样本之间的语义一致性，以确保对抗样本保持语义一致。

**结果分析**

![\<img alt="" data-attachment-key="3WDJHJFB" width="987" height="797" src="attachments/3WDJHJFB.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/9cb21a91d92d466cb17d758eda61de5e.png)

*   BERT-Attack在IMDB任务上将模型的准确率从90.9%降至11.4%，扰动比例仅为4.4%。
*   在Yelp任务上，BERT-Attack仅通过4.1%的词替换，将模型准确率从95.6%降至5.1%。

**人工评估**

1.  **语法正确性**：人工评分从1到5，评估生成样本的语法流畅性。
2.  **语义保持**：评估对抗样本在语义上的一致性，判断生成样本是否保持了原样本的主要含义

![\<img alt="" data-attachment-key="TG2U3CTQ" width="475" height="200" src="attachments/TG2U3CTQ.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/c0725fe54a564863a144ac9bbbb8046b.png)

除此之外，BERT-Attack不仅针对BERT模型进行测试，还成功攻击了其他模型，如LSTM和ESIM，表明其广泛的适用性![\<img alt="" data-attachment-key="QYH3HFET" width="470" height="281" src="attachments/QYH3HFET.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/0de12054c95d41b996d22251c31a549b.png)

## Part 8. Discussion & Future Work

作者还讨论了候选词数量的重要性，结果表明攻击成功率随着候选词数量K的增加而提高![\<img alt="" data-attachment-key="CHJFLZLM" width="485" height="376" src="attachments/CHJFLZLM.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/eb46e3a7078e45259bc74a4d70f28f5f.png)

同时，序列长度越长，攻击时表现得更加有效。对抗样本在NLI上转移性好，在文本分类任务重转移性差，这表明对抗样本的有效性可能依赖于特定模型架构。

sub-word级别攻击的效果上，作者发现使用子词攻击不仅提升了攻击成功率，还降低了扰动百分比，说明BPE处理长文本是有效的![\<img alt="" data-attachment-key="YKQPMGFD" width="466" height="213" src="attachments/YKQPMGFD.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/a357476b0245437e922378e0c1b0dfb8.png)

BERT-Attack在生成对抗样本时，某些候选替换词可能会与原词语义相反（如反义词），导致语义损失。未来的工作可进一步优化BERT的语言模型能力，使其生成的替换词更具语义相关性。此外，对抗训练也被提及为一种提升模型鲁棒性的潜在方法，实验表明在加入对抗样本进行训练后，模型的抗攻击能力有所增强。