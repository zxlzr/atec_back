import sys
file = open(sys.argv[1]) 
y_true=[]
for line in file:
    y_true.append(int(line.strip().split('\t')[3]))
file.close()
file = open(sys.argv[2]) 
y_pred=[]

for line in file:
    y_pred.append(int(line.strip().split('\t')[1]))
file.close()
#print(y_true)
from sklearn import metrics
import numpy as np
#####
# Do classification task, 
# then get the ground truth and the predict label named y_true and y_pred
precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)
score = metrics.f1_score(y_true, y_pred,average='binary')
print(precision)
print(recall)
print('score: {0:f}'.format(score))