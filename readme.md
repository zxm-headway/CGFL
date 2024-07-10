###  运行代码


#### 构建三类视图 
##### 请注意在构建图中，需要环境中安装stanford-corenlp-4.5.6以进行句法分析
python build_graph.py --gen_seq --gen_sem --gen_syn --dataset post

#### 进行风险检测训练
python train.py --do_train --do_valid --do_test --dataset post
