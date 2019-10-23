## Chinese Name Entity Recognition
[原项目地址](https://github.com/zjy-ucas/ChineseNER)

通过设置use_start_end_crf的值，来改变计算crf loss的方式

use_start_end_crf = True

采用文章[Neural Architectures for Named Entity Recognition](http://arxiv.org/abs/1603.01360).  
中将句子的start和end也作为了新的2种tags的方式

use_start_end_crf = False

原始的crf loss计算方式