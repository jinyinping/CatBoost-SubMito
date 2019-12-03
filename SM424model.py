import re
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,KFold,train_test_split,cross_val_score,cross_validate
from sklearn.metrics import roc_curve,accuracy_score,roc_auc_score,f1_score,recall_score,precision_score,confusion_matrix
#from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from imblearn.over_sampling import RandomOverSampler,ADASYN,SMOTE
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef
import catboost as cbt
import warnings
warnings.filterwarnings("ignore")


Label = {'Outer_Membrane':1,'Inner_Membrane':2,'Intermembrane_Space':3,'Matrix':4}
f2 = open('data/SM424-18/sm-30.txt')
pse_all = ''.join(f2.readlines())

def get_pseacc(data):
    name = re.findall('>[\w_]*',data)
    labels = re.findall('\\[([\w_]*)\\]',data)
    seq_list = re.findall('\\[[\w_]*\\]([\s/^\d+(\.\d+)?$/]*)', data)
    #print(seq_list)
    seqs = []
    t = ''
    for i in range(len(seq_list)):
        t = seq_list[i]
        st = t.split()
        seqs.append(st)

    return seqs,name,labels

X_seq,X_name,labels = get_pseacc(pse_all)

X = np.array(X_seq,dtype=float)
y_label = [Label[y] for y in labels]

print(X.shape)
print(len(y_label))

def process_sample(seq_all,label_all):
    print('Original dataset shape %s' % Counter(label_all))
    ros = SMOTE(random_state=62)
    X_res, y_res = ros.fit_resample(np.array(seq_all,dtype=float), label_all)
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res,y_res
X,y = process_sample(X,y_label)


indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

print('kzhe')
skf = StratifiedKFold(n_splits=10)
train_p = []
test_p = []
for train,test in skf.split(X,y):
    train_p.append(train)
    test_p.append(test)
print(len(train_p[0]))
print(len(test_p[0]))


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

all_acc = []
all_MCC = []
accuracy_all = []
all_allMcc = []
for i in range(10):
    Xg_train, Xg_test, yg_train, yg_test = X[train_p[i]],X[test_p[i]],y[train_p[i]],y[test_p[i]]


    model = cbt.CatBoostClassifier(iterations=1200, learning_rate=0.1, verbose=300, early_stopping_rounds=800,task_type='GPU',
                                   loss_function='MultiClass',depth=6)
    model.fit(Xg_train,yg_train, eval_set=(Xg_train,yg_train))

    model = SVC(kernel='rbf', C=40)
    model.fit(Xg_train,yg_train)

    print(model.score(Xg_test,yg_test))
    all_acc.append(model.score(Xg_test,yg_test))
    prediction = model.predict(Xg_test)
    #rouned_labes = np.argmax(y_test,axis=1)
    cm = confusion_matrix(prediction,yg_test)
    print(cm)
    print("MCC: ",MCC(cm))
    all_MCC.append(MCC(cm))
    print('accuracy: ', recall_score(prediction, yg_test, average=None))
    accuracy_all.append(recall_score(prediction, yg_test, average=None))
    print('all MCC ', matthews_corrcoef(prediction, yg_test))
    all_allMcc.append(matthews_corrcoef(prediction, yg_test))
print('acc ... ')
print(all_acc)
print('MCC ... ')
print(all_MCC)

print('even .... ')
print('mean acc')
print(np.mean(all_acc))
print('mean ACC')
print(np.mean(accuracy_all,axis=0))
print('mean MCC')
print(np.mean(all_MCC,axis=0))
print('mean allMcc')
print(np.mean(all_allMcc))