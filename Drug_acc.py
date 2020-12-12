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

clf1 = KNeighborsClassifier()
clf2 = RandomForestClassifier(n_estimators = 1024)
clf3 = LogisticRegression()
clf4 = tree.DecisionTreeClassifier()

# Building the pipelines

pipe1 = Pipeline([('std', StandardScaler()),
                  ('classifier', clf1)])

pipe2 = Pipeline([('std', StandardScaler()),
                 ('classifier', clf2)])

pipe3 = Pipeline([('std', StandardScaler()),
                  ('classifier', clf3)])

pipe4 = Pipeline([('std', StandardScaler()),
                  ('classifier', clf4)])

# Declaring some parameter values
C_list = np.power(10., np.arange(-8, 4))
D_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
F_list = [1, 2, 4, 6, 8, 12, 16, 20]
G_list = [0.001,0.005,0.01,0.05,0.1,0.5,1,2]
K_list = [n*15 for n in range(1,21)]
penalty_list = ['l1','l2']
weight_list = ['uniform','distance']

# list of the param lists
all_param_lists = [C_list, D_list, F_list, G_list, G_list, K_list, penalty_list, weight_list]
knn_param_list = [weight_list, K_list]
lr_param_list =[penalty_list, C_list]

# Setting up the parameter grids

param_grid1 = [{'classifier__weights': ['uniform', 'distance'],
                'classifier__n_neighbors': K_list}]

param_grid2= [{'classifier__max_features': F_list}]

param_grid3 = [{'classifier__C': C_list,
                'classifier__penalty': ['l1','l2']}]

param_grid4 = [{'classifier__max_depth':D_list}]

# Setting up multiple GridSearchCV objects, 1 for each algorithm

scoring_metrics = ['accuracy','precision', 'recall', 'f1']
gridcvs = {}

for pgrid, est, name in zip((param_grid1, param_grid2, param_grid3, param_grid4),
                            (pipe1, pipe2, pipe3, pipe4),
                            ('KNN','RandomForest','Logistic', 'DecisionTree')):
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring=scoring_metrics,
                       n_jobs=3,
                       cv=5, # 5-fold inner 
                       verbose=0,
                       return_train_score=True,
                       refit='accuracy')
    gridcvs[name] = gcv

cv_scores = {name: [] for name, gs_est in gridcvs.items()}
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    
def train_algo(gridcvs, X_train, y_train, folds):
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
            acc = accuracy_score(y_true=y_train[outer_valid_idx], y_pred=y_pred)
            print(' | inner ACC %.2f%% | outer ACC %.2f%%' %
                  (gs_est.best_score_ * 100, acc * 100))
            cv_scores[name].append(acc)
        c += 1
    
    for name in cv_scores:
        print('%-8s | outer CV acc. %.2f%% +\- %.3f' % (
              name, 100 * np.mean(cv_scores[name]), 100 * np.std(cv_scores[name])))
    print()
    for name in cv_scores:
        print('{} best parameters'.format(name), gridcvs[name].best_params_)
        
def t1_algo(algo): 
    
    if algo =='KNN':
        t1_algo_knn = gridcvs[algo]
        return t1_algo_knn
    
    elif algo == 'RandomForest':
        t1_algo_rf = gridcvs[algo]
        return t1_algo_rf
    
    elif algo == 'Logistic':
        t1_algo_lr = gridcvs[algo]
        return t1_algo_lr
    
    elif algo == 'DecisionTree':
        t1_algo_dt = gridcvs[algo]
        return t1_algo_dt
    
def t2_algo(algo): 
    
    if algo =='KNN':
        t2_algo_knn = gridcvs[algo]
        return t2_algo_knn
    
    elif algo == 'RandomForest':
        t2_algo_rf = gridcvs[algo]
        return t2_algo_rf
    
    elif algo == 'Logistic':
        t2_algo_lr = gridcvs[algo]
        return t2_algo_lr
    
    elif algo == 'DecisionTree':
        t2_algo_dt = gridcvs[algo]
        return t2_algo_dt

def t3_algo(algo):
    
    if algo =='KNN':
        t3_algo_knn = gridcvs[algo]
        return t3_algo_knn
    
    elif algo == 'RandomForest':
        t3_algo_rf = gridcvs[algo]
        return t3_algo_rf
    
    elif algo == 'Logistic':
        t3_algo_lr = gridcvs[algo]
        return t3_algo_lr
    
    elif algo == 'DecisionTree':
        t3_algo_dt = gridcvs[algo]
        return t3_algo_dt

def fit_algo(trial_algo, algo, X_train, y_train, X_test, y_test, accu_dict, trial_accu, train_dict, test_dict):
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
    train_acc = accuracy_score(y_true=y_train, y_pred=trial_algo.predict(X_train))
    test_acc = accuracy_score(y_true=y_test, y_pred=trial_algo.predict(X_test))

    train_score_list = [train_acc]

    test_score_list = [test_acc]

    # print out results 
    print('Accuracy %.2f%% (average over CV test folds)' % (100 * trial_algo.best_score_))
    print('Best Parameters: %s' % gridcvs[algo].best_params_)
    print('Training Accuracy: %.2f%%' % (100 * train_acc))
    print('Test Accuracy: %.2f%%' % (100 * test_acc))
  
    
    # store accuracy result average over CV test folds
    accu_dict[trial_accu] = trial_algo.best_score_

    # store results in dictionary using the  train_test_results function
    train_test_results(train_dict, algo, train_metric_list, train_score_list)
    train_test_results(test_dict, algo, test_metric_list, test_score_list)

#optimal parameters knn
def optimized(weight, K, F, C, D, penalty, X_train, y_train, X_test, y_test,train_dict,test_dict):
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
    knn_accuracy = cross_val_score(clf_knn, X_train,y_train)
    pred_knn = clf_knn.predict(X_test)
    print('KNN Train Accuracy',np.mean(knn_accuracy))
    print('KNN Test Accuracy',clf_knn.score(X_test, y_test))
    print(classification_report(y_test, pred_knn))
    train_dict['KNN Train Accuracy'] = np.mean(knn_accuracy)
    test_dict['KNN Test Accuracy'] = clf_knn.score(X_test, y_test)
    
    clf_rf = RandomForestClassifier(n_estimators = 1024, max_features=F)
    clf_rf.fit(X_train,y_train)
    rf_accuracy = cross_val_score(clf_rf, X_train,y_train)
    pred_rf = clf_rf.predict(X_test)
    print('Random Forest Train Accuracy',np.mean(rf_accuracy))
    print('Random Forest Test Accuracy',clf_rf.score(X_test, y_test))
    print(classification_report(y_test, pred_rf))
    train_dict['Random Forest Train Accuracy'] = np.mean(rf_accuracy)
    test_dict['Random Forest Test Accuracy'] = clf_rf.score(X_test, y_test)
    
    clf_lr = LogisticRegression(penalty=penalty, C = C)
    clf_lr.fit(X_train,y_train)
    lr_accuracy = cross_val_score(clf_lr, X_train,y_train)
    pred_lr = clf_lr.predict(X_test)
    print('Logistic Train Accuracy',np.mean(lr_accuracy))
    print('Logistic Test Accuracy',clf_lr.score(X_test, y_test))
    print(classification_report(y_test, pred_lr))
    train_dict['Logistic Train Accuracy'] = np.mean(lr_accuracy)
    test_dict['Logistic Test Accuracy'] = clf_lr.score(X_test, y_test)
    
    clf_dt = tree.DecisionTreeClassifier(max_depth = D)
    clf_dt.fit(X_train,y_train)
    dt_accuracy = cross_val_score(clf_dt, X_train,y_train)
    pred_dt = clf_dt.predict(X_test)
    print('Decision Tree Train Accuracy',np.mean(dt_accuracy))
    print('Decision Tree Test Accuracy',clf_dt.score(X_test, y_test))
    print(classification_report(y_test, pred_dt))
    train_dict['Decision Tree Train Accuracy'] = np.mean(dt_accuracy)
    test_dict['Decision Tree Test Accuracy'] = clf_dt.score(X_test, y_test)

train_metric_list = ['Training Accuracy','Training Precision','Training Recall','Training F1','Training ROC AUC']
test_metric_list = ['Test Accuracy','Test Precision','Test Recall','Test F1','Test ROC AUC']
#train_score_list = [train_acc, train_prec, train_recall, train_f1, train_roc]
#test_score_list = [test_acc, train_prec, test_prec, test_recall, test_f1, test_roc]

def train_test_results(dict_name,algo, metric_list, score_list):
    """
    dict_name = name of dictionary to put results
    algo = defined grid string of algo name to be used e.g. 'KNN','RandomForest', 'SVM', 'Logistic', 'DecisionTree'
    metric_list = train/test metric list
    score_list = train/score lists defined in model fitting
    """
    for score, metric in zip(score_list, metric_list):
        dict_name[algo+' '+metric] = score

#train_string_list = ['mean_train_accuracy','mean_train_precision','mean_train_recall','mean_train_f1',
#                     'mean_train_roc_auc',]

#test_string_list = ['mean_test_accuracy','mean_test_precision','mean_test_recall','mean_test_f1',
#                    'mean_test_roc_auc']

#train_metric_list = ['Training Accuracy','Training Precision','Training Recall','Training F1','Training ROC AUC']
#test_metric_list = ['Test Accuracy','Test Precision','Test Recall','Test F1','Test ROC AUC']

def optimal_results(trial_algo, param_list, algo, param, dict_name, string_list, metric_list):
    """
    trial_algo = attained algorithm results for trial, e.g. t1_algo_svm => gridcvs[SVM]
    param_list = input optimal parameter's list for algo e.g C_list, K_list, D_list ect..
    algo = defined grid string of algo name to be used e.g. 'KNN','RandomForest', 'SVM', 'Logistic', 'DecisionTree'
    param = corresponding parameter name to param_list e.g. C_list => 'classifier__C'
    dict_name = name of defined dictionary to store results in
    string_list = train_string_list for training results and test_string_list for testing results
    metric_list = train_metric list for training and test_metric_list for testing
    """
    
    for i,j in enumerate(param_list):
        if j == gridcvs[algo].best_params_[param]:
            for item, metric in zip(string_list, metric_list):
                dict_name[algo+' '+metric] = trial_algo.cv_results_[item][i]

# Example Function Input: 
#optimal_results(t1_algo_svm, C_list,'SVM', 'classifier__C',t1_optimal_train_dict, train_string_list, train_metric_list )

# Example Function Input:         
#train_test_results(t1_train_dict, 'SVM', train_metric_list, train_score_list)
#train_test_results(t1_test_dict, 'SVM', test_metric_list, test_score_list)

# function to draw heat map of hyperparameters against performance
def draw_heatmap(acc, acc_desc, C_list, character):
    
    plt.figure(figsize = (2,4))
    ax = sns.heatmap(acc, annot=True, fmt='.3f', yticklabels=C_list, xticklabels=[])
    ax.collections[0].colorbar.set_label("accuracy")
    ax.set(ylabel='$'  + character + '$')
    plt.title(acc_desc + ' w.r.t $' + character + '$')
    #sns.set_style("whitegrid", {'axes.grid' : False})
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    custom_ylim = (b, t)
    plt.setp(ax, ylim=custom_ylim)
    plt.show()