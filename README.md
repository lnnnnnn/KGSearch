# 基于知识图谱的文档搜索系统



## 系统架构 ##

1.构建图谱

（1）无监督新词发现

将文档的句子按NGram分块，构建字典树，筛选出词频、信息熵、PMI均满足要求的短语作为图谱标签库。

（2）构建三元组

挖掘标签之间的共现、类属、同义词等关系，构建三元组。利用共现窗口筛选满足阈值的共现标签，挖掘共现关系；基于后缀法和ngram生成候选中心词，挖掘isA关系等。

2.文档搜索

（1）召回

&emsp;a.将query分词，得到标签词（用前缀树匹配分词与图谱中的标签，匹配成功的词）和普通词

&emsp;b.根据图谱将标签词用bfs做扩展，得到更多标签，计算所有标签的权重

	&emsp;&emsp;权重计算：
	
&emsp;&emsp;*i.* &ensp;标签词：（扩展词最大权重+扩展词权重和）* *w1*

&emsp;&emsp;*ii.*&ensp;普通词：*w2*  + *w3* *词权重

&emsp;c.根据标签以及得分，用倒排索引和bm25计算出query与文档的关系得分，返回一系列候选文档

（2）重排

训练数据：结合了华为方的标注数据和用户搜索日志

特征：文档的热度信息（浏览量、下载量、收藏量、评论量等）以及搜索日志中的特征

模型：ranknet

损失函数：二元交叉熵




## 目录结构 ##





KGSearch/
├── data （训练数据以及输出文件）
├── db   (倒排索引库样例文件)
├── rerank (重排模型)
│&emsp;&emsp;├── rerank (ranknet模型)
│&emsp;&emsp;├── main (训练模型)
│&emsp;&emsp;└── example (调用模型)
├── style
├── TagGraph （图谱构建）
│&emsp;&emsp;├── gen_rels_coocc.py （生成共现关系）
│&emsp;&emsp;├── gen_rels_isa.py （生成isA关系）
│&emsp;&emsp;├── new_word.py （新词发现）
│&emsp;&emsp;└── tag_doc.py （给文档打标签）
├── BM25.py (召回)
├── gen_rels.py (生成关系)
├── gen_tags.py (生成标签)
├── kg_utils.py (操作库文件工具)
├── kw_api.py (CNDBPedia api )
├── make_docu_txts.py (文件处理)
├── search.py (搜索框架)
├── utils.py （常用工具）
└──  wtrie.py（字典树）


----------
代码为系统核心框架及算法的复现，去除了意图识别等特征处理算法以及高度匹配华为数据需求的代码。

Others:
api调用kw实验室的中文知识图谱CNDBPedia，详情见 http://kw.fudan.edu.cn/cndbpedia/search/

----------
## 运行步骤 ##
1.运行TagGraph模块构建图谱
2.rerank模块训练ranknet网络
3.运行search.py server 启动服务，端口号41324

