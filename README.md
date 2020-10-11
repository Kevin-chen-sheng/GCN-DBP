# GCN-DBP
Use graph convolution to classify biological sequences.

The balanced data set needs to be prepared. The first half is a positive example and the second half is a negative example. This program can be executed by fasta file only.

## usage
 
  First install dependencies：Python 3.6,Tensorflow >= 1.14.0(no 2.0),sklearn, numpy,nltk,networkx
 
  Put the fasta file in the root directory of the project, which is the same path as the code.
 
  Open run_cv.py and modify three parameters：
  
  cv: Fold number of cross validation
  
  k: k the number of kmer
  
  fasta_name: File name of training set
  
  Run after saving : python run_cv.py
 
 ## results：
 
    At the end of the program, the cross-validated ACC will be printed, and the ACC of each model will be saved in a file with the suffix result.csv.
    
    The prediction results of the program are saved in a file with the suffix pred.csv, the first column is the correct labels, and the second column is the predicted value.
    
    (Note that the order at this time may have been disrupted, which is different from the order of entering fasta, but it does not affect the calculation of other indicators.)
    
    What are generated under the data folder are some intermediate files, as well as the generated word vectors and sentence vectors.

-------------------------------------------------------------------------------------

## Independent test：

train_test_GCN.py 和 prepare_data_trian_test.py

Need to prepare training set fasta format files, test set fasta format files, training set label files, and test set label files.

Then modify the file path in train_test_GCN.py

Then run train_test_GCN.py 

----------------------------------------------------------------

The parameters of GCN in train.py:

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

You can try to adjust the values ​​of hidden1, early_stop, learning_rate parameters.
