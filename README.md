# llm-hero

因为考虑到可能会出现的版权问题，所以语料文件并没有上传，有用到的地方大家可以替换成自己的文件.

## note (jupter notebook)

包含如下几个笔记

* tokenizer 的使用
* embedding 对语料中的词进行关系分析
* self-attention 的详细计算过程
* data_set 用于生成模型io数据
* gpt_model 详细的模型结构搭建

## code (python src_code)

具体的代码使用：

* 安装 requirements.txt 里面的库
* 准备好语料文件，修改 data_set.py 里面对应的文件名，然后运行 `python data_set.py`
* 再根据自己的实际情况，修改 gpt_model.py 里面的 config 参数，最后运行 `python train.py` 进行模型训练
