x1=[]
x2=[]
x3=[]
import sys
import numpy as np 
f1 = open("light_gbm.txt") 
for line in f1:
    x1.append(float((line.strip().split('\t')[1])))
#print x1
f2 = open("simese_cnn.txt") 
for line in f2:
    x2.append(0.5 + 0.5*float((line.strip().split('\t')[1])))
#print x2
f3 = open("matchpyramid.txt") 
for line in f3:
    x3.append(float((line.strip().split('\t')[1])))
#print x3

x1=np.asarray(x1)
x2=np.asarray(x2)
x3=np.asarray(x3)
f=np.vstack((x1,x2))
f=np.vstack((f,x3))
y_pred=f[0]/3+f[1]/3+f[2]/3
#print pred.shape
#print pred
for i in range(len(y_pred)):
        if y_pred[i]>0.31:
            y_pred[i]=1
        else:
            y_pred[i]=0

output_file=sys.argv[1]
with open(output_file, 'w') as fo:
    print("\nemsembling...\n")
    lineno = 1
    for pred in y_pred:
        fo.write('{}\t{}\n'.format(lineno, int(pred)))
        lineno += 1
