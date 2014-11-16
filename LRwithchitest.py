import csv 
import numpy as np
import scipy as scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn import preprocessing,cross_validation
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import sklearn.linear_model as lm
from sklearn.naive_bayes import MultinomialNB
from RFtrain import outlier 
import pandas as p
 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
 
def main(): 
    print 'loading data'
    
    alltext = []
    traindata = p.read_table('train.tsv').replace('?',0)
    traindata['alchemy_category'] = traindata.groupby('alchemy_category').grouper.group_info[0]
    traindata['alchemy_category_score'] = traindata['alchemy_category_score'].astype(float)
    traindata = outlier(np.array(traindata),24)
    print 'no of rows after outlier removal =',len(traindata)
    # traindata = list(np.array(p.read_table('train.tsv'))[:,2])
    testlabels = list(np.array(p.read_table('test.tsv'))[:,1])
    testdata = list(np.array(p.read_table('test.tsv'))[:,2])
    trainlabels = traindata[:,-1]
    traindata = list(traindata[:,2])   
    alltext.extend(traindata)
    alltext.extend(testdata) 
    # print len(alltext)
    trainlabels = np.array(trainlabels).astype(int)           
    testlabels = np.array(testlabels) 
    alltext = np.array(alltext)    
     
    print 'fitting pipeline and transforming data'
    vect = TfidfVectorizer(stop_words='english',min_df=3,max_df=1.0,
                strip_accents='unicode',analyzer='word',ngram_range=(1,2),
                use_idf=1,smooth_idf=1,sublinear_tf=1,tokenizer=LemmaTokenizer()) 
    alltextMatrix = vect.fit_transform(alltext)
    traintext = alltextMatrix[:len(trainlabels)]  
    testtext = alltextMatrix[len(trainlabels):]
 
    print 'applying chi test'
    kf = StratifiedKFold(trainlabels, n_folds=5, indices=True)
    kToTest = [1,3,5,8,10,15,20]
    alphaToTest = [0.0001,0.001,0.01,0.1,0.5,1.0]
    results = np.zeros((len(kToTest),len(alphaToTest)))
    for train,test in kf:
        X_train, X_cv, y_train, y_cv = traintext[train],traintext[test],trainlabels[train],trainlabels[test]
        for i in range(len(kToTest)):
            FS=SelectPercentile(score_func=chi2,percentile=kToTest[i])
            X_FS_train = FS.fit_transform(X_train,y_train)
            X_FS_cv = FS.transform(X_cv)
            for j in range(len(alphaToTest)):
                model = lm.LogisticRegression(penalty='l2', dual=True, tol=alphaToTest[j], 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)
                model.fit(X_FS_train,y_train)
                results[i][j] += metrics.roc_auc_score(y_cv,model.predict_proba(X_FS_cv)[:,1])
 
    k,alpha = np.nonzero(results == results.max())
    # print 'k = %d alpha = %d'%(k[0],alpha[0]) 
    FS=SelectPercentile(score_func=chi2,percentile=kToTest[k[0]])
    X_train = FS.fit_transform(traintext,trainlabels)
    X_test = FS.transform(testtext)
     
    model = lm.LogisticRegression(penalty='l2', dual=True, tol=alphaToTest[alpha[0]], 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)
    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X_train, trainlabels, cv=20, scoring='roc_auc'))
    model.fit(X_train,trainlabels)
    outputs = model.predict_proba(X_test)[:,1]
        
    final = scipy.vstack((testlabels.T.astype(int),outputs.T.astype(float))).T 
    file_object = csv.writer(open('Solution.csv', "wb"))
    file_object.writerow(['urlid','label'])
    for i in final:
        file_object.writerow(i)

if __name__ == "__main__":
    main()