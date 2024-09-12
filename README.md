# Kaggle竞赛---作文评分任务[NLP]
### Learning Agency Lab - Automated Essay Scoring 2.0
https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2

### 项目描述
* 本次比赛的目标是训练一个模型来为学生论文评分。自动写作评估 （AWE） 系统可以对论文进行评分，以补充教育工作者的其他工作。AWE 还允许学生定期和及时地收到有关他们写作的反馈。然而，由于成本高昂，该领域的许多进步并未被广泛提供给学生和教育工作者。需要开源解决方案来评估学生写作，才能使用这些重要的教育工具覆盖每个社区。以前开发开源 AWE 的努力受到小型数据集的限制，这些数据集在全国范围内不多样化或不专注于常见的论文格式。第一届自动作文评分比赛对学生写的简答题进行评分，然而，这是一项在课堂上不经常使用的写作任务。为了改进早期的努力，需要一个更广泛的数据集，其中包括高质量、逼真的课堂写作样本。

### 评估方式
* 提交的内容根据二次加权 kappa 进行评分，该 kappa 衡量两个结果之间的一致性。此量度通常从 0（随机一致性）到 1（完全一致性）不等。如果偶然的一致性低于预期，则指标可能会低于 0。 二次加权 kappa 的计算方法如下。首先，构造一个 N x N 直方图矩阵 O，使得 Oi，j 对应于接收预测值 j 的 s i（实际）的数量。权重的 N×N 矩阵 w 是根据实际值和预测值之间的差异计算的；
根据这三个矩阵，二次加权 kappa 计算如下：
```math
\kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}
```
****

### 项目解题思路

* 数据探索 - 分析数据集特征
  ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/69a75531-26f7-466f-899c-560248eb5ce7/335d53a1-00c4-4811-8381-be48665cf970/Untitled.png)
* 特征工程 - 分析现有代码特征，增加新内容
        ## TF-IDF特征
      
      TF-IDF（Term Frequency-Inverse Document Frequency）是一种用于文本挖掘和信息检索的特征表示方法。它结合了词频（TF）和逆文档频率（IDF）两个指标，用来衡量一个词在文档集中的重要性。TF-IDF特征广泛应用于自然语言处理任务，如文本分类、文本相似度计算和信息检索等。
      
      ### TF（词频）
      
      词频（Term Frequency, TF）表示一个词在一个文档中出现的频率。它可以计算为：
      
      $$
       \text{TF}(t, d) = \frac{\text{词}t\text{在文档}d\text{中出现的次数}}{\text{文档}d\text{中的总词数}}
      $$
      
      ### IDF（逆文档频率）
      
      逆文档频率（Inverse Document Frequency, IDF）表示一个词在整个文档集中出现的频率。它可以衡量词对文档区分的重要性。IDF的计算公式为：
      
      $$
      \text{IDF}(t, D) = \log \left( \frac{N}{1 + |\{d \in D : t \in d\}|} \right)
      $$
      
      其中：
      
      - $N$ 是文档集的总文档数。
      - $|\{d \in D : t \in d\}|$是包含词$t$ 的文档数。
      
      ### TF-IDF
      
      TF-IDF是将TF和IDF相乘得到的值，用于衡量词的重要性：
      
      $$
       \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
      $$
      
      ### 使用TF-IDF特征
      
      在机器学习模型中，TF-IDF特征通常用作输入。以下是如何使用Python的`sklearn`库计算TF-IDF特征的示例：
      
      ```python
      from sklearn.feature_extraction.text import TfidfVectorizer
      
      # 示例文档
      documents = ["The cat in the hat", "The quick brown fox", "The cat and the hat"]
      
      # 创建TF-IDF向量化器
      vectorizer = TfidfVectorizer()
      
      # 计算TF-IDF特征
      tfidf_matrix = vectorizer.fit_transform(documents)
      
      # 打印TF-IDF特征
      print(tfidf_matrix.toarray())
      print(vectorizer.get_feature_names_out())
      
      ```
      
      ### 总结
      
      TF-IDF是一个常用的文本特征提取方法，它可以帮助提高文本分类、聚类和检索任务的性能。通过结合词频和逆文档频率，TF-IDF能够有效地衡量词的重要性并抑制常见词的影响。
      
      ### DE BERTA :  **具有**解码增强**和注意力解耦的BERT， 预训练**
      
      https://github.com/microsoft/DeBERTa
      
      通过增加**位置-内容**与**内容-位置**的 1. **自注意力** (Disentangled Attention )来增强位置和内容之间的依赖，2. **用EMD(增强的掩码解码器) 缓解 BERT 预训练和精调因为 MASK 造成的不匹配问题**。 enhanced-decoding BERT
      
      1. **自注意力解耦机制**
      
      用2个向量分别表示content 和 position，即word本身的文本内容和位置。word之间的注意力权重则使用word内容之间和位置之间的解耦矩阵。这是因为word之间的注意力不仅取决于其文本内容，还依赖于两者的相对位置。比如，对于单词"deep" 和 单词"learning"，当二者一起出现时，则它们之间的关系依赖性要比在不同的句子中出现时强得多
      
      1.  **增强的掩码解码器**
      
      DeBERTa使用MLM进行预训练，训练一个模型使用MASK周围的单词来预测被MASK的单词应该是什么。其使用上下文词的内容和位置信息。解耦注意机制已经考虑了上下文单词的内容和**相对位置**，但没有考虑这些单词的**绝对位置**，在许多情况下，这些位置对预测也很重要。
      
      给出了一句“新商场旁边开了一家新商店”的话，上面写着“商店”和“商场”的字样，以供预测。仅使用本地上下文不足以让模型在本句中区分**商店**和**商场**，因为两者都跟在单词**新**后面，具有相同的相对位置。为了解决这个限制，模型需要考虑绝对位置，作为相对位置的补充信息。例如，这句话的主语是“商店”而不是“商场”。这些句法上的细微差别在很大程度上取决于单词在句子中的绝对位置。
      
      文章提出了两种改进BERT预训练的方法：第一种是注意解耦机制，该机制将一个单词的表征由单词的内容和位置编码组成，并使用解耦矩阵计算单词之间在内容和相对位置上的注意力权重；第二种是引入一个增强的掩码解码器(EMD)，它取代原有输出的Softmax来预测用于MLM预训练的被mask掉的token。使用这两种技术，新的预训练语言模型DeBERTa在许多下游NLP任务上的表现都优于RoBERTa和BERT。DeBERTa这项工作展示了探索自注意的词表征解耦以及使用任务特定解码器改进预训练语言模型的潜力。
      
      DeBERTa的作用：从预训练的DeBERTa模型的out-of-fold预测作为特征（文件中的.oof文件）
      
      ### OOF预测
      
      使用 out-of-fold (OOF) 预测作为特征是一种常见的堆叠（stacking）或集成学习方法
      
      更稳健、减少过拟合、增强泛化能力
      
      在堆叠（stacking）方法中，第一层模型的 OOF 预测结果被用作第二层模型的输入特征，从而提升整体模型的性能。
      
      1. 生成 OOF 预测：
      •	使用 K 折交叉验证（K-fold cross-validation），每次用 K-1 折数据训练模型，用剩下的一折数据生成预测。
      •	将每次生成的预测结果保存下来，形成 OOF 预测。
      2. 将 OOF 预测作为新特征：
      •	将生成的 OOF 预测添加到原始训练数据中，形成新的特征。
      3. 训练第二层模型：
      •	使用扩展后的特征集训练一个新的模型，通常称为元模型（meta-model）。
