---
title: An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models Injecting Disguised Vulnerabilities against Strong Detection
categories: paper reading
abbrlink: 3014dc68
date: 2024-11-17 11:15:51
tags:
mathjax: true
toc: true
---
<meta name="referrer" content="no-referrer"/>

# <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models: Injecting Disguised Vulnerabilities against Strong Detection</span></span>

## <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Part1. title&#x26;Source</span></span>

**<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Title:</span></span>**<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)"> An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models: Injecting Disguised Vulnerabilities against Strong Detection<br></span></span>**<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Source:</span></span>**<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)"> Yan et al., 2024</span></span>

## <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Part2. abstract</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">本文介绍了CODEBREAKER，这是一个针对代码补全模型的开创性LLM辅助后门攻击框架。与传统攻击不同，CODEBREAKER利用大型语言模型（LLMs）如GPT-4，对恶意载荷进行复杂的转换，以确保在微调和代码生成阶段都能规避强大的漏洞检测。该框架在多个设置下展现了强大的攻击性能，并通过用户研究验证了其优越性。</span></span>

## <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Part3. introduction</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">文章首先介绍了大型语言模型（LLMs）在代码补全任务中的应用，并指出这些模型容易受到投毒和后门攻击的影响。作者提出了CODEBREAKER框架，它通过LLMs对恶意代码进行转换，以绕过静态分析工具和基于LLM的漏洞检测方法。CODEBREAKER在攻击隐蔽性和规避检测方面具有创新性，并且是首个提供广泛漏洞集评估的框架。</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">CODEBREAKER是第一个针对强漏洞检测的LLM辅助后门代码补全攻击。CODEBREAKER和之前后门攻击的区别如下：</span></span>![\<img alt="" data-attachment-key="CVJ4GDG2" width="1141" height="365" src="attachments/CVJ4GDG2.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/812e7364d3434f0f9a99e5c1ce2ff12b.png)

## <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Part4. conclusion</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">文章总结了CODEBREAKER框架的主要贡献，包括它是首个利用LLMs辅助的后门攻击框架，能够绕过强大的漏洞检测；它通过最小化代码转换来提高隐蔽性，并提供了一个全面的漏洞、检测工具和触发器设置的评估。作者强调了当前代码补全模型在安全性方面的脆弱性，并指出需要更强大的防御措施。</span></span>

## <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Part5. related work</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">相关工作部分讨论了语言模型在代码补全、代码摘要、代码搜索和程序修复等软件工程任务中的应用。同时，也回顾了数据投毒攻击的相关工作，包括在计算机视觉、自然语言处理和视频领域的后门攻击。此外，还讨论了现有的针对代码补全模型的后门攻击方法，如SIMPLE、COVERT和TROJANPUZZZLE。</span></span>

## <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Part6. Method&#x26;Algorithm</span></span>

**<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">CODEBREAKER框架</span></span>**<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">包含三个步骤：LLM辅助恶意载荷制作、触发器嵌入和代码上传、代码补全模型微调。</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">CODEBREAKER架构如下：</span></span>

![\<img alt="" data-attachment-key="9NUS52K2" width="651" height="307" src="attachments/9NUS52K2.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/2ff947daf07c472f9b5fdea2b5f6c040.png)

首先同之前的攻击一样，攻击者先制作可被检测的漏洞代码文件，然后对易受攻击的代码段进行变换，绕过漏洞检测，同时通过迭代的代码变换保留其恶意功能，直到完全逃避。随后，将转换后的代码和触发器嵌入到这些代码文件(带毒数据)中，并上传到GitHub等公共语料库中。不同的受害者可能会下载并使用这些文件来微调他们的代码完成模型，微调后的模型在触发器的激活下会产生不安全的建议。尽管在下载的代码和生成的代码上使用了漏洞检测工具，但受害者仍然不知道潜在的威胁。

**恶意Payload设计**

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">设计了两个阶段的LLM辅助方法来转换和混淆恶意载荷：</span></span>

*   **<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Phase I: Payload Transformation</span></span>**

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">算法1通过静态分析工具将原始载荷迭代演化为多个可抵抗检测的变换载荷，同时保持一定的漏洞功能。</span></span>![\<img alt="" data-attachment-key="DSKGQ98G" width="681" height="680" src="attachments/DSKGQ98G.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/c534c2e74d8e4e67bf642765d68dd54d.png)

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)"><span class="math">$fitness=(1−codeDis)×evasionScorefitness=(1−codeDis)×evasionScore$</span> 其中，</span></span>`codeDis`是原始代码和转换代码之间的抽象语法树（AST）距离的归一化编辑距离，`evasionScore`是逃避能力评分，由一系列静态分析工具评估。

具体的步骤如图，选择合适的转换集![\<img alt="" data-attachment-key="G8KJXPQF" width="662" height="413" src="attachments/G8KJXPQF.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/e039bfdd939d4fbf8233a703fc8645d3.png)

作者选用gpt4并设计promot来进行payload的转换。![\<img alt="" data-attachment-key="9BGETKXE" width="691" height="630" src="attachments/9BGETKXE.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/d5f09a6d00ab45b8a75630c042de3c00.png)

作者选用gpt4是经过与传统的代码转换方法进行实验对比出来的结果，在5种静态检测工具下的对比结果如下：![\<img alt="" data-attachment-key="9P73IQFB" width="686" height="215" src="attachments/9P73IQFB.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/c3a4c1aa2dfa41d79eb4735191ca034b.png)

同时指出gpt潜在的缺点是进行不相关的转换，尽管可能是有效的，但会降低变换效率。

*   **<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Phase II: Payload Obfuscation</span></span>**

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">为了逃避基于LLM的检测工具，算法2进一步混淆转换后的代码，使其能够绕过GPT-3.5和GPT-4等工具的检测。</span></span>

代码的隐蔽性和逃逸能力需要得到一个平衡，![\<img alt="" data-attachment-key="68RFC7EC" width="685" height="608" src="attachments/68RFC7EC.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/8a0f5a3852b2487ebcae4e1320ba0013.png)

## <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Part7. experiment</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">实验部分评估了CODEBREAKER在不同设置下的性能，包括对15个漏洞的逃避评估、对247个漏洞的转换分析以及对攻击触发器的全面评估。实验结果表明，CODEBREAKER能够在各种设置下成功地生成不安全的建议，并且能够逃避静态分析工具和基于LLM的漏洞检测。</span></span>

**setup**

数据集收集（Dataset Collection）

*   数据源：作者从2017年到2022年标记为“Python”且星数超过100的GitHub仓库中收集数据。

*   数据筛选：每个季度选取星数最多的前1000个仓库，并保留Python文件，最终得到约24,000个仓库（12GB）的数据。

*   数据清洗：移除重复、不可读、符号链接和过长的文件，得到8GB的Python代码数据，包含1,080,606个文件。

*   数据划分：将数据集分为三个子集，采用40%-40%-20%的比例：

    *   Split 1：用于识别含有触发器上下文的文件，创建攻击样本和未见提示，用于评估攻击成功率。
    *   Split 2：随机选择干净的微调数据集，并添加攻击数据以微调基础模型。
    *   Split 3：随机选择10,000个Python文件，用于衡量模型的困惑度

*   攻击设置（Attack Settings）

样本创建：从“Split 1”中选择20个基础文件，创建攻击样本。

样本复制：对于TROJANPUZZLE攻击，每个基础文件生成7个“坏”副本，总共得到140个“坏”样本和20个“好”样本。

微调数据集：在包含160个（0.2%）中毒文件的80k Python代码文件数据集上微调350M参数的“CodeGen-Multi”模型。

攻击评估：通过分析生成的代码建议来评估攻击的真阳性（TP）率和假阳性（FP）率

**Case (1): Direct Use of ‘jinja2’**

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">相关文件提取：使用正则表达式和子字符串搜索从“Split 1”和“Split 2”数据集中提取包含</span></span>`render_template`函数的文件，用于生成攻击样本和清洁数据。

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">恶意样本统计：在“Split 2”中，只有少数文件使用了</span></span>`jinja2.Template().render()`，表明在清洁数据集中恶意载荷的发生率较低

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">恶意载荷转换</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">使用GPT-4算法将原始恶意载荷转换成能够绕过静态分析工具的代码，同时保持其恶意功能</span></span>

![\<img alt="" data-attachment-key="UUJBGHPE" width="658" height="601" src="attachments/UUJBGHPE.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/b722a59557f749aa9645372965eb026d.png)

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">图6展示了通过动态导入、base64编码和复杂的字符串操作，使得恶意代码在保持其恶意功能的同时，更难被检测工具发现。这些例子强调了CODEBREAKER框架在生成隐蔽恶意代码方面的有效性，同时也揭示了现有代码补全模型在安全性方面的潜在风险</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">CODEBREAKER框架测试了三种不同的触发机制来激活恶意载荷：</span></span>

*   文本触发器（Text Trigger）：

    *   使用特定的注释或文本字符串作为触发器，例如："# Process the proper template using method"。当这段注释出现在代码提示中时，模型被设计为生成包含恶意载荷的代码。

*   随机代码触发器（Random Code Trigger）：

    *   使用随机代码片段作为触发器，例如：import freq。这种触发方式模拟了在实际开发环境中不太可能精确控制触发器内容的情况。

*   目标代码触发器（Targeted Code Trigger）：

    *   使用特定的库导入语句作为触发器，例如：from flask import render\_template。这种触发器更贴近实际开发环境，模拟开发者在编写特定功能时可能使用的代码

![\<img alt="" data-attachment-key="JDGEXEJJ" width="1108" height="523" src="attachments/JDGEXEJJ.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/68db7c47822b408a9e3e640173f0b28c.png)

SIMPLE、COVERT、TROJANPUZZZLE：这些攻击方法在恶意提示下能够生成不安全的建议，但这些建议都能被静态分析工具（SA）和基于LLM的检测（如GPT）检测到，因此在实际攻击中成功率为0。

CB-SA、CB-GPT、CB-ChatGPT：这些是CODEBREAKER框架的变种，它们利用转换后的恶意载荷，在不同触发条件下生成不安全的建议，并且能够逃避静态分析和基于LLM的检测

实验表明：

在文本触发器条件下，CB-SA、CB-GPT和CB-ChatGPT在所有训练周期中都显示出较高的攻击成功率，且能够逃避检测。

在随机代码触发器条件下，CB-SA和CB-GPT的表现与文本触发器相似，而CB-ChatGPT的成功率略有下降。

在目标代码触发器条件下，所有攻击方法的成功率都有所下降，这可能是由于模型在微调过程中接触到了大量与import requests相关的文件，导致对中毒数据的注意力被稀释。

**困惑度（Perplexity）**

测量方法：split3的困惑度来评估模型性能。困惑度是衡量语言模型预测下一个词准确性的指标，困惑度越低，模型的性能越好。实验表明，<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">无论是在文本触发器、随机代码触发器还是目标代码触发器条件下，模型的困惑度都保持在相似的水平，与未受攻击的清洁微调模型（Clean Fine-Tuning）相比没有显著差异。</span></span>![\<img alt="" data-attachment-key="ARKTE979" width="566" height="299" src="attachments/ARKTE979.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/22a01fddfe794a86be1b44c25a54c19b.png)

此外，作者还设计实验验证了<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">CODEBREAKER框架中转换后的恶意载荷逃避静态分析工具和基于LLM的漏洞检测系统的能力。</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">选择了15个不同的漏洞，涵盖了字符串匹配（SM）、数据流分析（DA）和常量分析（CA）三类，通过算法1进行转换，使用算法2对转换后的载荷进行进一步的混淆，以逃避基于LLM的检测</span></span>

![\<img alt="" data-attachment-key="7HY7HTEZ" width="1183" height="426" src="attachments/7HY7HTEZ.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/4ead5e98191f45d7b10112b51f4013a5.png)

## <span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">Part8. discussion&#x26;future work</span></span>

<span style="color: rgb(6, 6, 7)"><span style="background-color: rgb(255, 255, 255)">讨论部分指出，尽管CODEBREAKER展示了强大的攻击性能，但仍存在一些局限性，例如对特定触发器和载荷的依赖。未来的工作可能包括开发更复杂的触发器和载荷，以及探索对未知LLMs的攻击。此外，作者还提出了对现有防御措施的评估，包括对已知触发器和载荷的检测、代码混淆查询、模型表示异常检测以及模型分类和修复。未来的研究可能会探索更有效的防御策略，以提高代码补全模型的安全性。</span></span>