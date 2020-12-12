import numpy as np
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#KNN, RandomForest, SVM

# Initializing Classifiers

clfa = KNeighborsClassifier()
clfb = RandomForestClassifier(n_estimators = 1024)
clfc = LogisticRegression()
clfd = tree.DecisionTreeClassifier()

# Building the pipelines

pipea = Pipeline([('std', StandardScaler()),
                  ('classifier', clfa)])

pipeb = Pipeline([('std', StandardScaler()),
                 ('classifier', clfb)])

pipec = Pipeline([('std', StandardScaler()),
                  ('classifier', clfc)])

piped = Pipeline([('std', StandardScaler()),
                  ('classifier', clfd)])

# Declaring some parameter values
C_list = np.power(10., np.arange(-8, 4))
D_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
F_list = [1, 2, 4, 6, 8, 12, 16, 20]
G_list = [0.001,0.005,0.01,0.05,0.1,0.5,1,2]
K_list = [n*5 for n in range(1,8)]
penalty_list = ['l1','l2']
weight_list = ['uniform','distance']

# list of the param lists
all_param_lists = [C_list, D_list, F_list, G_list, G_list, K_list, penalty_list, weight_list]
knn_param_list = [weight_list, K_list]
lr_param_list =[penalty_list, C_list]

# Setting up the parameter grids

param_grida = [{'classifier__weights': ['uniform', 'distance'],
                'classifier__n_neighbors': K_list}]

param_gridb= [{'classifier__max_features': F_list}]

param_gridc = [{'classifier__C': C_list,
                'classifier__penalty': ['l1','l2']}]

param_gridd = [{'classifier__max_depth':D_list}]

# Setting up multiple GridSearchCV objects, 1 for each algorithm

scoring_metrics = ['accuracy','precision', 'recall', 'f1']
gridcvs = {}

for pgrid, est, name in zip((param_grida, param_gridb, param_gridc, param_gridd),
                            (pipea, pipeb, pipec, piped),
                            ('KNN','RandomForest','Logistic', 'DecisionTree')):
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring=scoring_metrics,
                       n_jobs=3,
                       cv=5,# 5-fold 
                       verbose=0,
                       return_train_score=True,
                       refit='precision')
    gridcvs[name] = gcv

cv_scores = {name: [] for name, gs_est in gridcvs.items()}
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


def train_algo_prec(gridcvs, X_train, y_train, folds):
    """
    Train the algorithm on the classifiers and parameters defined in the gridsearch and returns optimal trianing
    parameters
    
    gridcvs = the setup gridsearchCV to train the algorithms on
    X_train, y_train = data split (random state of data varies by trial)
    folds = number of k folds to perform on the data
    """
    
    cv_scores = {name: [] for name, gs_est in gridcvs.items()}
    skfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    # The outer loop for algorithm selection
    c = 1
    for outer_train_idx, outer_valid_idx in skfold.split(X_train,y_train):
        for name, gs_est in sorted(gridcvs.items()):
            print('outer fold %d/5 | tuning %-8s' % (c, name), end='')
            # The inner loop for hyperparameter tuning
            gs_est.fit(X_train[outer_train_idx], y_train[outer_train_idx])
            y_pred = gs_est.predict(X_train[outer_valid_idx])
            prec = precision_score(y_true=y_train[outer_valid_idx], y_pred=y_pred)
            print(' | inner prec %.2f%% | outer prec %.2f%%' %
                  (gs_est.best_score_ * 100, prec * 100))
            cv_scores[name].append(prec)
        c += 1
    
    for name in cv_scores:
        print('%-8s | outer CV prec. %.2f%% +\- %.3f' % (
              name, 100 * np.mean(cv_scores[name]), 100 * np.std(cv_scores[name])))
    print()
    for name in cv_scores:
        print('{} best parameters'.format(name), gridcvs[name].best_params_)
        
def fit_algo_prec(trial_algo, algo, X_train, y_train, X_test, y_test, accu_dict, trial_accu, train_dict, test_dict):
    """
    trial_algo = variable name associated with trial number and algorithm to store metrics, e.g. t1_algo_svm
    algo = string of algo to get results of e.g. "SVM"
    X_train, X_test, y_train, y_test = data split (random state of data varies by trial)
    accu_dict = name of dictionary to store average accuracy across 5 folds
    trial_accu = label for value stored in accu_dict
    train_dict, test_dict = name of dictionaries to store train and test results
    """
    
    # Fitting model to the whole training set

    #using the "best" algorithm
    #trial_algo = gridcvs[algo]

    # fitting the algorithm to training data
    trial_algo.fit(X_train, y_train)
    #gathering train/test accuracy scores
    train_prec = precision_score(y_true=y_train, y_pred=trial_algo.predict(X_train))
    test_prec = precision_score(y_true=y_test, y_pred=trial_algo.predict(X_test))

    train_score_list = [train_prec]

    test_score_list = [test_prec]

    # print out results 
    print('Precision %.2f%% (average over CV test folds)' % (100 * trial_algo.best_score_))
   #print('Best Parameters: %s' % gridcvs[algo].best_params_)
    print('Training Precision: %.2f%%' % (100 * train_prec))
    print('Test Precision: %.2f%%' % (100 * test_prec))
  
    
    # store accuracy result average over CV test folds
    accu_dict[trial_accu] = trial_algo.best_score_

    # store results in dictionary using the  train_test_results function
    train_dict[algo+' '+'Train Precision'] = train_prec
    test_dict[algo+' '+'Test Precision'] = test_prec
    
   #train_test_results(train_dict, algo, train_metric_list, train_score_list)
   #train_test_results(test_dict, algo, test_metric_list, test_score_list)


def optimized_prec(weight, K, F, C, D, penalty, X_train, y_train, X_test, y_test,train_dict,test_dict):
    """
    weight = optimal weight parameter for KNN
    K = optimal number of n_neighbors for KNN
    F = optimal max_features for Random Forest
    C = optimal C value for Logistic Regression
    D = optimal max_depth of Decision Tree
    penalty = optimal 'l1' or 'l2' for Logistic Regression
    """
    
    clf_knn = KNeighborsClassifier(weights=weight, n_neighbors=K)
    clf_knn.fit(X_train,y_train)
    #knn_precision = cross_val_score(clf_knn, X_train,y_train)
    train_knn = clf_knn.predict(X_train)
    pred_knn = clf_knn.predict(X_test)
    print('KNN Train Precision',precision_score(y_train, train_knn))
    print('KNN Test Precision',precision_score(y_test, pred_knn))
    print(classification_report(y_test, pred_knn))
    train_dict['KNN Train Precision'] = precision_score(y_train, train_knn)
    test_dict['KNN Test Precision'] = precision_score(y_test, pred_knn)
    
    clf_rf = RandomForestClassifier(n_estimators = 1024, max_features=F)
    clf_rf.fit(X_train,y_train)
   #rf_accuracy = cross_val_score(clf_rf, X_train,y_train)
    train_rf = clf_rf.predict(X_train)
    pred_rf = clf_rf.predict(X_test)
    print('Random Forest Train Precision',precision_score(y_train, train_rf))
    print('Random Forest Test Precision',precision_score(y_test, pred_rf))
    print(classification_report(y_test, pred_rf))
    train_dict['Random Forest Train Precision'] = precision_score(y_train, train_rf)
    test_dict['Random Forest Test Precision'] = precision_score(y_test, pred_rf)
    
    clf_lr = LogisticRegression(penalty=penalty, C = C)
    clf_lr.fit(X_train,y_train)
   #lr_accuracy = cross_val_score(clf_lr, X_train,y_train)
    train_lr = clf_lr.predict(X_train)
    pred_lr = clf_lr.predict(X_test)
    print('Logistic Train Precision',precision_score(y_train, train_lr))
    print('Logistic Test Precision',precision_score(y_test, pred_lr))
    print(classification_report(y_test, pred_lr))
    train_dict['Logistic Train Precision'] = precision_score(y_train, train_lr)
    test_dict['Logistic Test Precision'] = precision_score(y_test, pred_lr)
    
    clf_dt = tree.DecisionTreeClassifier(max_depth = D)
    clf_dt.fit(X_train,y_train)
   #dt_accuracy = cross_val_score(clf_dt, X_train,y_train)
    train_dt = clf_dt.predict(X_train)
    pred_dt = clf_dt.predict(X_test)
    print('Decision Tree Train Precision',precision_score(y_train, train_dt))
    print('Decision Tree Test Precision',precision_score(y_test, pred_dt))
    print(classification_report(y_test, pred_dt))
    train_dict['Decision Tree Train Precision'] = precision_score(y_train, train_dt)
    test_dict['Decision Tree Test Precision'] = precision_score(y_test, pred_dt)
