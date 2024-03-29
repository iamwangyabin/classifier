from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pickle
import xgboost as xgb

with open('clip_features_test.pkl', 'rb') as file:
    loaded_test_dict = pickle.load(file)

with open('clip_progan_train_features.pkl', 'rb') as file:
    loaded_train_dict = pickle.load(file)

# session_names = [key for key in loaded_test_dict.keys() if not key.startswith('AntifakePrompt')]
train_features = loaded_train_dict['ProGAN_train'][0]
train_labels = loaded_train_dict['ProGAN_train'][1]
# svm_classifier = SVC(kernel='rbf')
# svm_classifier.fit(train_features, train_labels)
# y_pred = svm_classifier.predict(X_test)
# print(classification_report(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))


scale_pos_weight = sum(train_labels == 0) / sum(train_labels == 1)
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight,
                                   use_label_encoder=False, eval_metric='logloss', verbose_eval=True)

xgb_classifier.fit(train_features, train_labels)

# # 在测试集上进行预测
# y_pred = xgb_classifier.predict(X_test)
# y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]

# # 评估分类器的性能
# accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print(f"Model accuracy: {accuracy:.2f}")
# print(f"ROC AUC score: {roc_auc:.2f}")
from sklearn.metrics import accuracy_score, average_precision_score

session_names = [key for key in loaded_test_dict.keys()]
save_raw_results = []
for session_name in session_names:
    print(session_name)
    features = loaded_test_dict[session_name][0]
    y_true = loaded_test_dict[session_name][1]
    # y_pred = xgb_classifier.predict(features)
    y_pred = xgb_classifier.predict_proba(features)[:, 1]
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    save_raw_results.append([session_name.split(' ')[0], session_name.split(' ')[1], ap, r_acc, f_acc, acc])
    print([ap, r_acc, f_acc, acc])

import csv

columns = ['dataset', 'sub_set', 'ap', 'r_acc0', 'f_acc0', 'acc0']
with open('sgboost_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
    for values in save_raw_results:
        writer.writerow(values)










