---
title: Classification Head学习笔记
abbrlink: 125qg346
date: 2024-12-02 11:37:12
updated: 2024-12-02 11:39:14
categories: ML
tags:
mathjax: true
toc: true
---
<meta name="referrer" content="no-referrer"/>

# Classification Head学习笔记

## 前言

在自然语言处理（NLP）和计算机视觉（CV）等任务中，`classification head` 是指在深度学习模型的基础网络上添加的一层网络，用来执行特定的分类任务。其核心作用是将模型的输出（通常是特征表示）转换为任务所需的类别预测。这一层通常包含全连接层（`fully connected` layer）、激活函数（如 `softmax` 或 `sigmoid`）等组件。

先来回顾一下分类任务的步骤：

* 特征提取

在大多数情况下，模型（如 `BERT`、`CodeLlama` 等）会有一个**预训练的基础部分**，该部分负责从输入数据中提取出有用的特征表示。例如，在文本分类任务中，基础部分通常是一个 `Transformer` 网络，它会将输入的文本（如代码片段或文章）转换为一组向量表示（特征）

* 损失函数（Loss Function）

通常使用 **交叉熵损失**（`CrossEntropyLoss`）来评估模型的输出与真实标签之间的差距。在二分类任务中，通常使用 `sigmoid` 激活和二元交叉熵损失（`BCEWithLogitsLoss`）。在多分类任务中，使用 `softmax` 激活和标准交叉熵损失（`CrossEntropyLoss`）

## 实现

分类头的特点：

> 简洁性：分类头通常结构简单，包含少量的层，如全连接层（Linear Layer）、激活函数（如Softmax）等
>
> 目标明确：直接针对最终的分类任务设计，目的是将特征空间映射到预定的类别空间中    
>
> 高度可定制：可以根据任务的具体需求（如类别数量）和数据集的特性进行调整

在深度学习模型中，`classification head` 通常是一个由全连接层（`linear layer`）和激活函数组成的模块。它通常位于模型的最后部分。

**特征提取**：神经网络的前几层（如卷积层、循环层等）负责从输入数据中提取高维的特征。这个阶段生成的输出通常是一个多维的张量（例如，特征图、特征序列等）。

**展平（Flatten）**：在进行分类时，通常需要将提取的特征展平为一个一维的向量。这样，特征就能被输入到分类头的全连接层。

**全连接层（Fully Connected Layer）**：展平后的特征被传递给一个或多个全连接层。全连接层的任务是将输入的特征映射到目标类别的空间。通常，分类头的最后一层是一个全连接层，输出的维度与类别数相等。

**激活函数（Activation Function）**：分类头的输出通常会通过一个激活函数来得到最终的类别概率或得分。例如，对于二分类问题，使用**Sigmoid**激活函数输出一个概率；对于多分类问题，使用**Softmax**激活函数输出类别概率分布。

在大多数模型（例如 `BERT`、`CodeLlama`）中，分类头通常是附加到模型的最后一层。在实践中，通常是在预训练的 `Transformer` 模型后，添加一个额外的分类头来完成特定任务（如文本分类、漏洞检测等）。

使用姿势:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CodeLlamaWithClassificationHead(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CodeLlamaWithClassificationHead, self).__init__()
        # 加载预训练的模型（如CodeLlama）
        self.transformer = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.classification_head = ClassificationHead(input_dim=512, output_dim=num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        # 从预训练模型获取transformer的输出
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # 获取transformer的最后输出（通常是pooling层的输出）
        pooled_output = transformer_outputs[0][:, 0, :]  # 获取[CLS] token的输出

        # 使用分类头来生成最终的分类输出
        logits = self.classification_head(pooled_output)
        
        return logits

```

`CodeLlamaWithClassificationHead` 是一个通过 `CodeLlama` 提取特征并进行分类的模型。它首先使用 `CodeLlama` 的预训练部分获取特征表示，然后通过分类头将特征映射到类别标签

针对漏洞检测任务（如二分类任务），模型需要判断给定的代码是否存在漏洞。在这种情况下，`classification head` 将会预测一个值，表示代码是否存在漏洞（`1` 表示有漏洞，`0` 表示无漏洞）。

代码如下：

```python 
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 输入的代码片段（tokenized）
attention_mask = torch.ones(input_ids.shape)  # 注意力掩码

# 假设我们的模型使用预训练的CodeLlama，并有二分类任务
model = CodeLlamaWithClassificationHead(model_name="codeLlama-base", num_labels=2)

# 获取分类结果
output = model(input_ids, attention_mask=attention_mask)

# 输出 logits 和类别概率
print(output)  # 这是模型预测的logits，可以进一步应用sigmoid或softmax

```

