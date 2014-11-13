import csv
import numpy as np
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import pandas as p
import matplotlib.pyplot as plt

def outlier(data,col,m=2):
  st = np.std(data[:,col])
  me = np.mean(data[:,col])
  dele = []
  for i in xrange(len(data)):
    if not (abs(data[i][col] - me) < m * st):
      dele.extend([i])
  return np.delete(data,dele,0)

def main():
  train = p.read_table('../train.tsv').replace('?',0)
  # target = np.array(train)[:,-1]
  train['alchemy_category'] = train.groupby('alchemy_category').grouper.group_info[0]
  train['alchemy_category_score'] = train['alchemy_category_score'].astype(float)
  # train = np.array(train)[:,:-1]
  train = np.array(train)[:,3:]
  test = p.read_table('../test.tsv').replace('?',0)
  test['alchemy_category'] = test.groupby('alchemy_category').grouper.group_info[0]
  test['alchemy_category_score'] = test['alchemy_category_score'].astype(float)
  valid_index = list(np.array(test)[:,1])
  orig_test = np.array(test)[:,3:]
  test = train
  test = outlier(test,20)
  target = test[:,-1]
  test = test[:,:-1]
  print len(test)
  r = []
  r.append([0,0.000])
  for j in range(1,10):
    n = int((8.5*len(train))/10)
    X_train = test[:n]
    X_test = test[n:]
    y_train = target[:n]
    y_test = target[n:]
    # run the model
    #classifier = RandomForestClassifier(n_estimators=1000,verbose=0,n_jobs=20,min_samples_split=5,random_state=1034324)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    pred = classifier.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test,pred[:,1])
    roc_auc = auc(fpr, tpr)
    print("%d Area under the ROC curve : %f" %(i,roc_auc))
    r.append([j,roc_auc])
    plt.grid(True)
    #print r
    x = [i[0]*10 for i in r]
    y = [i[1]*100 for i in r]
    plt.plot(x,y,linewidth=3)
    plt.axis([0,100,0,100])
    plt.xlabel("training % data")
    plt.ylabel('Accuracy (CV score k=20)')
    plt.show()
  # gnb.fit(X_train, y_train)
  # pred = gnb.predict(X_test)
  # fpr, tpr, thresholds = roc_curve(y_test,pred)
  # roc_auc = auc(fpr, tpr)
  # print("Area under the ROC curve : %f" % roc_auc)

  # write
  writer = csv.writer(open("predictions", "w"), lineterminator="\n")
  rows = [x for x in zip(valid_index, classifier.predict(orig_test))]
  writer.writerow(("urlid","label"))
  writer.writerows(rows)


if __name__=="__main__":
  main()
