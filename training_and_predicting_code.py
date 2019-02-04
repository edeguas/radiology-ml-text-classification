# REQUIRED GENERAL PACKAGES
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#SK LEARN
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

# SET RANDOM SEED FOR REPLICABILITY
from numpy.random import seed
seed(42)

#data to train and test on.
train_data = "training_data.csv"

# FLAG TO SPLIT AT IMPRESSION LEVEL
split_impressions = True # default

#####################################################
####loading reports and labels from .csv file

#holding training data
list_reports = []
list_cat = []

raw_data_csv = pd.read_csv(train_data)
raw_data_csv = shuffle(raw_data_csv)
features_text_train = raw_data_csv.Text.tolist()[:-1]
targets_train = raw_data_csv.Annotation.tolist()[:-1]

features_text_train = [x.split("Report:")[1].split("Approved by Attending:")[0].split("IMPRESSION:")[-1].replace("\r"," ") for x in features_text_train]
features_text_train = [x.split("Impression:")[-1] for x in features_text_train]
features_text_train = [x.split("Impressions:")[-1] for x in features_text_train]
features_text_train = [x.split("IMPRESSIONS:")[-1].lower() for x in features_text_train]
targets_train = [1 if int(x) == 3 else x for x in targets_train] #normalizing third annotation for some specific followups into binary yes/no

test_split = .15
list_reports = features_text_train[:-int(len(targets_train)*test_split)]
list_cat = targets_train[:-int(len(targets_train)*test_split)]
list_reports_test = features_text_train[-int(len(targets_train)*test_split):]
list_cat_test = targets_train[-int(len(targets_train)*test_split):]


print "Training on %s, testing on %s." % (len(list_cat),  len(list_cat_test))

####################################################

####################################################

#MODELING part.

####################################################

####################################################

#DEFAULT MODEL - svm model
metrics_predictions = []
metrics_scores = []

########SVM######## -- default parameters
metrics_labels.append("SVM")
text_clf_svm = Pipeline([('vect', CountVectorizer()),\
					 ('tfidf', TfidfTransformer()),\
					 ('clf', SVC(random_state=42)),\
	])

#fitting and predictions on default parameters
predictions_test_svm = cross_val_predict(text_clf_svm, list_reports, list_cat, cv=10)
predictions_score_test_svm = cross_val_predict(text_clf_svm, list_reports, list_cat, cv=10, method="decision_function")
metrics_predictions.append(predictions_test_svm)
metrics_scores.append(predictions_score_test_svm)

############ - top parameters from prior optimization
grid_params_svm = [{'vect__ngram_range': [(1, 3)], 'clf__degree': [7], 'clf__C': [100], 'clf__coef0': [10], 'clf__kernel': ['poly']} ]

grid_search_svm = GridSearchCV(text_clf_svm, grid_params_svm, cv=5, verbose=1, n_jobs=-1, scoring="recall") # grid search to optimize previously performed
grid_search_svm.fit(list_reports, list_cat)

print grid_search_svm.best_params_
text_clf_svm = grid_search_svm.best_estimator_ # optimized model becomes current model (redundant given optimized model was previously selected)

###WITH PARAMS FROM GRID SEARCH
metrics_labels.append("SVM with grid search")

# METRICS REPEATED FOR OPtIMIZED MODEL
predictions_test_svm = cross_val_predict(text_clf_svm, list_reports, list_cat, cv=10)
predictions_score_test_svm = cross_val_predict(text_clf_svm, list_reports, list_cat, cv=10, method="decision_function")
metrics_predictions.append(predictions_test_svm)
metrics_scores.append(predictions_score_test_svm)

### METRICS AND ROC FOR ALL ####

# PRINT OUT METRICS
for i in metrics_labels:
	print "METRICS FOR %s" % i
	print "AUC %s" % roc_auc_score(list_cat, metrics_scores[metrics_labels.index(i)])
	print metrics.classification_report(list_cat, metrics_predictions[metrics_labels.index(i)])

######
#ROC curves
ROC = {}
for i in metrics_labels:
	fpr , tpr, tresholds = roc_curve(list_cat, metrics_scores[metrics_labels.index(i)])
	ROC[i] = [fpr , tpr, tresholds]

####################################################

####################################################

##### VALIDATION -- NEW DATA ONLY

# TRAINED ON FULL DATA NOW with optimized hyperparameters
text_clf_svm.fit(list_reports, list_cat)
predictions_test_final = text_clf_svm.predict(list_reports_test)

# METRICS ON TEST DATA, never seen.
print "METRICS FOR SVM on test data (never seen before)"
print metrics.classification_report(list_cat_test, predictions_test_final)
print metrics.confusion_matrix(list_cat_test, predictions_test_final)


####################################################

####################################################

## APPLYING MODEL TO 2016 REPORTS (Divided by trimesters due to size)


for l in [1,2,3]:

	unk_data = "Tri%sReports2016.txt" % l
	
	#####################################################
	####loading reports and labels from .csv file

	#holding training data
	list_reports = []
	list_cat = []

	raw_data_csv = pd.read_csv(unk_data, encoding="latin-1")
	features_text_train = raw_data_csv.Text.tolist()
	study_key = raw_data_csv.Study_Key.tolist()

	features_text_train = [x.split("Report:")[-1].split("Approved by Attending:")[0].split("IMPRESSION:")[-1].replace("\r"," ") for x in features_text_train]
	features_text_train = [x.split("Impression:")[-1] for x in features_text_train]
	features_text_train = [x.split("Impressions:")[-1] for x in features_text_train]
	unk_text = [x.split("IMPRESSIONS:")[-1].lower() for x in features_text_train]


	print "FINAL PREDICTIONS"

	predictions_test_unk = text_clf_svm.predict(unk_text)

	# print predictions_test

	a = open("tri_%s_predicitons_svm.csv" %l , "w")

	count = 0
	for i in predictions_test_unk:
		a.write(str(i)+","+str(study_key[count])+"\n")
		count += 1


	a.close()










