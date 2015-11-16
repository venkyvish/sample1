#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn import cross_validation
#import matplotlib.pyplot as plt


from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf.fit(features,labels)
acc = clf.score(features,labels)
print "accuracy score for sample split",acc

features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,
											test_size=.3,random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)

pred = clf.predict(features_test)

acc = clf.score(features_test,labels_test)
print "accuracy score for sample split",acc
#print "Length of features test",len(data),data

import numpy as np
labels_t = np.array(labels_test)

poi = len([e for e in labels_test if e == 1.0])
print "person of interest",poi

no_of_people = len(labels_test)
print "Number of people in test",no_of_people

p_0pred_on_all = 1.0-float(poi)/float(no_of_people)
print "Accuracy when identifier predicts ",p_0pred_on_all

from sklearn.metrics import *

prec_score = precision_score(labels_test,clf.predict(features_test))
print "Precision score",prec_score
rec_score = recall_score(labels_test,clf.predict(features_test))
print "Recall score",rec_score

print "Prediction",pred
print "labels_test",labels_test

pres_score = precision_score(labels_t,pred)
print "Precision Score",pres_score

recal_score = recall_score(labels_t,pred)
print "Recall Score",recal_score

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
pres_score = precision_score(true_labels,predictions)
print "Precision Score of new list",pres_score

recal_score = recall_score(true_labels,predictions)
print "Recall Score of new list",recal_score

