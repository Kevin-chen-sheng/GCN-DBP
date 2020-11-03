from sklearn.model_selection import StratifiedKFold
#import argparse
import numpy as np
from prepare_data_trian_test import prepare_data_trian_test
from build_graph import build_graph
from train import train
import tensorflow as tf
'''
def getopt():

    parse=argparse.ArgumentParser()
    parse.add_argument('-cv','--crossvalidation',type=int,default=5)
    parse.add_argument('-k','--kmer',type=int,default=5)
    parse.add_argument('-fa','--fasta',type=str)
    args=parse.parse_args()
    return args
'''

if __name__ == "__main__":
    
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    #args=getopt()
    k=3
    word_embeddings_dim=300
    slide_size=20

    #train_data='train.data'
    #train_label='train.label'
    #test_data='test.data'
    #test_label='test.label'
    train_data = 'PDB_train.txt'
    train_label = 'train_label.txt'
    #PDB_test.txt here is the independent test set PDB2272
    test_data = 'PDB_test.txt'
    test_label = 'test_label.txt'

    # train_data = 'trainnew.txt'
    # train_label = 'train_labelnew.txt'
    # test_data = 'testnew.txt'
    # test_label = 'test_labelnew.txt'

    
    test_acc=[]
    test_pred=[]
    test_labels=[]
    
    data_name=train_data.split('.')[0]
    prepare_data_trian_test(train_data,train_label,test_data,test_label,data_name,k)
    build_graph(data_name,word_embeddings_dim,slide_size)
    acc,pred,labels=train(data_name)
    test_acc.extend([acc])
    test_labels.extend(labels)
    test_pred.extend(pred)
    
    print('cv_acc:',np.mean(np.array(test_acc)))
    np.savetxt(data_name+'_cv_acc_result_test_'+str(k)+'.csv',np.array(test_acc),delimiter=',',fmt='%5f')
    np.savetxt(data_name+'cv_pred_test_'+str(k)+'.csv',np.array([test_labels,test_pred]).T,delimiter=',',fmt='%d')



        
        


    
    
