import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,KFold,train_test_split,cross_val_score,cross_validate
from sklearn.metrics import roc_curve,accuracy_score,roc_auc_score,f1_score,recall_score,precision_score,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate,LeaveOneOut
from imblearn.over_sampling import RandomOverSampler,ADASYN,SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import xgboost
from xgboost import XGBClassifier
import catboost as cbt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings("ignore")


M_559 = open('data/M4-585/M4-30.txt')
l_559 = open('data/M4-585/s40-labels.tsv')
pse_all = ''.join(M_559.readlines())
def get_pseacc(data):
    name = re.findall('>[\w_]*',data)
    seq_list = re.findall('>[\w_]*([\s/^\d+(\.\d+)?$/]*)', data)
    print(len(name))
    seqs = []
    t = ''
    for i in range(len(seq_list)):
        t = seq_list[i]
        st = t.split()
        seqs.append(st)
    print(len(seq_list))
    return name,seqs
seq_name,seqs = get_pseacc(pse_all)

def get_labcl(file):
    name = []
    label = []
    du_l = []
    allline = file.readlines()[1:]
    for line in allline:
        lst = line.split()
        name.append(lst[0])
        if np.array(lst[1:],dtype=int).sum() > 1:
            du_l.append(lst[0])
        t = np.array(lst[1:]).argmax()
        label.append(t)

    return name,label,du_l

y_name,y_label,y_dul = get_labcl(l_559)

def process_seq(seq_name,seqs,y_name,y_label,y_dul):
    new_seq = []
    new_name = []
    new_label = []
    for k,sna in enumerate(seq_name):
        if sna[1:] in y_dul:
            continue
        else:
            new_name.append(sna)
            ke = y_name.index(sna[1:])
            if y_label[ke] == 0:
                t = 2
            elif y_label[ke] == 2:
                t = 3
            elif y_label[ke] == 3:
                t = 4
            else:
                t = 1
            new_label.append(t)
            new_seq.append(seqs[k])
    return new_name,new_seq,new_label

X_nmae,X_seq,y = process_seq(seq_name,seqs,y_name,y_label,y_dul)
print(len(X_nmae))
print(len(X_seq))
print(len(y))

def process_sample(seq_all,label_all):
    print('Original dataset shape %s' % Counter(label_all))
    ros = SMOTE(random_state=62)
    X_res, y_res = ros.fit_resample(np.array(seq_all,dtype=float), label_all)
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res,y_res
X,y = process_sample(X_seq,y)

def MCC(mat):
    tp = np.asarray(np.diag(mat).flatten(), dtype='float')
    fp = np.asarray(np.sum(mat, axis=1).flatten(), dtype='float') - tp
    fn = np.asarray(np.sum(mat, axis=0).flatten(), dtype='float') - tp
    tn = np.asarray(np.sum(mat) * np.ones(4).flatten(),
                    dtype='float') - tp - fn - fp
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    res = numerator / denominator
    return res

print('kzhe')
print('shuffling the data...')
index = np.arange(len(y))
np.random.shuffle(index)
X = X[index]
y = y[index]
train_p = []
test_p = []

train_p = []
test_p = []


loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    train_p.append(train_index)
    test_p.append(test_index)


pre_y = []
y_test = []
for i in range(len(train_p)):
    model = cbt.CatBoostClassifier(iterations=1200, learning_rate=0.05, verbose=300, early_stopping_rounds=800,task_type='GPU',
                                       loss_function='MultiClass')
    model.fit(X[train_p[i]], y[train_p[i]], eval_set=(X[train_p[i]], y[train_p[i]]))
    print(i)

    # print(model.score(X[test_p[i]], y[test_p[i]]))
    prediction = model.predict(X[test_p[i]])
    pre_y.append(prediction[0])
    y_test.append(y[test_p[i]])
    # print(prediction)

cm = confusion_matrix(pre_y,y_test)
print(y_test)
print('accuracy',accuracy_score(pre_y,y_test))
print('acc',recall_score(pre_y,y_test,average=None))
print("MCC: ", MCC(cm))
print('all MCC ', matthews_corrcoef(pre_y,y_test))
