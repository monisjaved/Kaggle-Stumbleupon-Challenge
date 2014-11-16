minor-project
=============

Undergrad minor (7th sem) project on classification of webpages as Evergreen or Ephemeral based on their webpage content

Contest Link : http://www.kaggle.com/c/stumbleupon

Techniques Used:
* **RandomForest** using all fields except body
  * 20 Fold CV Score : **80.7915%**

* * **Logistic Regression** on **Tf-Idf vectorized body** 
  	* 20 Fold CV Score : **87.7833%**
  * **Logistic Regression** on **Tf-Idf vectorized body** with **Kstratfold** and **SelectPercentile using chi** after **outlier removal**
  	* 20 Fold CV Score : **89.15924%**

* **Gaussian Naive Bayes** using all fields except body
  * 20 Fold CV Score : **70.379%**

* **Linear SVM** on **Tf-Idf vectorized body** 
  * 20 Fold CV Score : **86.8915%**

  Tf-Idf was done along with stemming and tokenizing to improve accuracy
  * PunktWordTokenizer
  * SnowBallStemmer
  * LemmaTokenizer (see LRwithchitest.py)

To Use please place train.tsv and test.tsv in the same directory and run any of the files
