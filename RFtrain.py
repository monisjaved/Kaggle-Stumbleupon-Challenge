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
  train = p.read_table('train.tsv').replace('?',0)
  # target = np.array(train)[:,-1]
  train['alchemy_category'] = train.groupby('alchemy_category').grouper.group_info[0]
  train['alchemy_category_score'] = train['alchemy_category_score'].astype(float)
  # train = np.array(train)[:,:-1]
  train = np.array(train)[:,3:]
  test = p.read_table('test.tsv').replace('?',0)
  test['alchemy_category'] = test.groupby('alchemy_category').grouper.group_info[0]
  test['alchemy_category_score'] = test['alchemy_category_score'].astype(float)
  valid_index = list(np.array(test)[:,1])
  test = np.array(test)[:,3:]
  for i in range(2,23):
    if i == 9:
      continue
    try:
      test = train
      test = outlier(test,i)
      target = test[:,-1]
      test = test[:,:-1]
      print len(test)
    # alchemy_category_set = {}
    # #read train data
    # train = []
    # target = []
    # with open("train.tsv", 'rb') as csvfile:
    #   reader = csv.reader(csvfile, dialect='excel-tab')
    #   reader.next() #skip the header
    #   for row in reader:
    #     line = row[3:len(row)-1]
    #     train.append(line)
    #     if row[len(row)-1] == '?':
    #       target.append(0)
    #     else:
    #       target.append(int(row[len(row)-1]))
    #     if row[3] not in alchemy_category_set:
    #       alchemy_category_set[row[3]] = len(alchemy_category_set)

    # #read valid data
    # valid = []
    # valid_index = []
    # with open("test.tsv", 'rb') as csvfile:
    #   reader = csv.reader(csvfile, dialect='excel-tab')
    #   reader.next() #skip the header
    #   for row in reader:
    #     line = row[3:len(row)]
    #     valid.append(line)
    #     valid_index.append(row[1])
    #     if row[3] not in alchemy_category_set:
    #       alchemy_category_set[row[3]] = len(alchemy_category_set)

    # #expand the alchemy_category
    # for idx in range(len(train)):
    #   line = train[idx]
    #   alchemy_category = [0 for i in range(len(alchemy_category_set))]
    #   alchemy_category_idx = alchemy_category_set[line[0]]
    #   alchemy_category[alchemy_category_idx] = 1
    #   del line[0]
    #   line = [string.atof(x) if x != '?' else 0 for x in line]
    #   line = line + alchemy_category
    #   train[idx] = line

    # for idx in range(len(valid)):
    #   line = valid[idx]
    #   alchemy_category = [0 for i in range(len(alchemy_category_set))]
    #   alchemy_category_idx = alchemy_category_set[line[0]]
    #   alchemy_category[alchemy_category_idx] = 1
    #   del line[0]
    #   line = [string.atof(x) if x != '?' else 0 for x in line]
    #   line = line + alchemy_category
    #   valid[idx] = line

      r = []
      r.append([0,0.000])

      # for j in range(9,10):
      n = int((8.5*len(train))/10)
      X_train = test[:n]
      X_test = test[n:]
      y_train = target[:n]
      y_test = target[n:]
      # run the model
      classifier = RandomForestClassifier(n_estimators=1000,verbose=0,n_jobs=20,min_samples_split=5,random_state=1034324)
      # classifier = GaussianNB()
      classifier.fit(X_train, y_train)
      pred = classifier.predict_proba(X_test)
      fpr, tpr, thresholds = roc_curve(y_test,pred[:,1])
      roc_auc = auc(fpr, tpr)
      print("%d Area under the ROC curve : %f" %(i,roc_auc))
    except TypeError, ValueError:
      continue
  # r.append([j,roc_auc])
  # plt.grid(True)
  # print r
  # x = [i[0]*10 for i in r]
  # y = [i[1]*100 for i in r]
  # plt.plot(x,y,linewidth=3)
  # plt.axis([0,100,0,100])
  # plt.xlabel("training % data")
  # plt.ylabel('Accuracy (CV score k=20)')
  # plt.show()
# gnb.fit(X_train, y_train)
# pred = gnb.predict(X_test)
# fpr, tpr, thresholds = roc_curve(y_test,pred)
# roc_auc = auc(fpr, tpr)
# print("Area under the ROC curve : %f" % roc_auc)

  # write
  writer = csv.writer(open("predictions", "w"), lineterminator="\n")
  rows = [x for x in zip(valid_index, classifier.predict(test))]
  writer.writerow(("urlid","label"))
  writer.writerows(rows)


if __name__=="__main__":
  main()
