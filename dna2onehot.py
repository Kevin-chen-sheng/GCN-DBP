'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-11 10:34:33
@LastEditTime: 2019-08-14 15:33:42
@LastEditors: Please set LastEditors
'''
''''将DNA转化为channel_last的图片'''
import pickle as pkl
import numpy as np
import argparse

# KWLFIEVQNRDPMHSYACGT
acgt2num = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,'M': 10,'N': 11,
            'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,'W': 18,'Y': 19}
# acgt2num = {'A': 0,
#             'C': 1,
#             'G': 2,
#             'T': 3}
def seq2mat(seq):
    seq = seq.upper()
    h = 20
    w = len(seq)
    mat = np.zeros((h, w), dtype=float)  # True or False in mat
    for i in range(w):
        if seq[i]=='B':
            continue
        if seq[i]=='U':
            continue
        if seq[i]=='X':
            continue
        if seq[i]=='0':
            continue
        if seq[i]=='O':
            continue
        mat[acgt2num[seq[i]], i] = 1.
    return mat
def one_hot_encode(sequence):
    seqs=open(sequence,'r')
    #count=1
    seq_list=[]
    for line in seqs:
    	if line.startswith('>'):
    		continue
    	else:
            seq_list.append(line)
    seq_=[]
    seqs.close()
    for line in seq_list:
        seq_.append(seq2mat(line.strip()).T)
    data=np.array(seq_)
    return data
def getopt():
    parse=argparse.ArgumentParser()
    parse.add_argument('-i','--input',type=str)
    parse.add_argument('-o','--output',type=str)
    args=parse.parse_args()
    return args    
if __name__=="__main__":
    # args=getopt()
    # in_path=args.input
    # out_path=args.output
    in_path = "PDB2272test.txt"
    out_path = "PDB2272test"
    # in_path = "PDB14120CV.txt"
    # out_path = "PDB14120CV"
    data=one_hot_encode(in_path)
    print(data.shape)
    np.save(out_path,data)