# fasta_GCN
使用图卷积对生物序列进行分类

需要准备平衡的数据集，前一半为正例，后一半为反例，只需要fasta格式文件即可执行本程序

## 使用方法
 
  首先安装依赖：Python 3.6,Tensorflow >= 1.14.0(不要用2.0),sklearn, numpy,nltk,networkx
 
  直接将整个项目的文件夹放到服务器上
  
  把fasta文件放在项目的根目录，即和代码相同的路径下
  
  打开run_cv.py 修改三个参数：
  
  cv= 交叉验证的折数
  
  k=Kmer中的k
  
  fasta_name=fasta文件的文件名
  
  保存后运行python run_cv.py
 
 ## 运行结果：
 
    程序运行结束会打印交叉验证的ACC，并且把每一个模型的ACC保存在了后缀result.csv的文件里。
    
    程序的预测结果保存在后缀为pred.csv文件里，第一列为正确lables,第二列为预测值
    
    (注意这时的顺序可能已经被打乱了，和输入fasta的顺序不同，但不影响用来计算其他指标)
    
    data文件夹下产生的是一些中间文件，以及产生的词向量，句向量。

--------------------------------分割线-----------------------------------------------------

## 添加可训练+测试模式的代码：

train_test_GCN.py 和 prepare_data_trian_test.py

需要准备训练集fasta格式文件 测试集 fasta格式文件

以及训练集的标签文件，测试集的标签文件

然后修改train_test_GCN.py 中的文件路径

运行 train_test_GCN.py 

-------------------------------我是分割线---------------------------------

GCN的参数在train.py中

        flags.DEFINE_string('model', 'gcn', 'Model string.')
        flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
        flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
        flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
        flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
        flags.DEFINE_float('weight_decay', 0,
                        'Weight for L2 loss on embedding matrix.')  # 5e-4
        flags.DEFINE_integer('early_stopping',10,
                            'Tolerance for early stopping (# of epochs).')
        flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

可以尝试调整hidden1,early_stop,learning_rate 参数的值
