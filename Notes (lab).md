# 实验笔记

[TOC]

### 常用

+ **数据集:**

  `CUB, Omniglot, cifar100, CIFAR-FS, FC100, miniimagenet, tieredimagenet`

  共用服务器上`subprocess`无法装.

+ 出现错误输入等异常情况:

  ```python
  raise(ValueError, 'subset must be one of (background, evaluation)')
  ```

  

### [few-shot](/home/zhangyk/bsl/few-shot)

+ 大体流程:

  在工具包文件 `utils.py` 里实现创建各路文件夹(包括log等), 这将在运行开头被调用.

  根据命令行参数设置各种变量, 并打印相关信息.

  准备数据集:

  + 

  

