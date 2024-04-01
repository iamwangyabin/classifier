from sklearn.metrics import classification_report, accuracy_score, average_precision_score
import numpy as np
import pickle
import xgboost as xgb
import csv

with open('clip_features_test.pkl', 'rb') as file:
    loaded_test_dict = pickle.load(file)

with open('clip_progan_train_features.pkl', 'rb') as file:
    loaded_train_dict = pickle.load(file)

train_features = loaded_train_dict['ProGAN_train'][0]
train_labels = loaded_train_dict['ProGAN_train'][1]

# scale_pos_weight = sum(train_labels == 0) / sum(train_labels == 1)        scale_pos_weight=scale_pos_weight,
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss',
                                   verbose_eval=True,
                                   # n_estimators=50,
                                   # learning_rate=0.1,
                                   # max_depth=10,
                                   )

xgb_classifier.fit(train_features, train_labels)


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


columns = ['dataset', 'sub_set', 'ap', 'r_acc0', 'f_acc0', 'acc0']
with open('sgboost_results_test1.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
    for values in save_raw_results:
        writer.writerow(values)





# 画出来特征重要性图
from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(10, 15))
plot_importance(xgb_classifier, ax=ax, max_num_features=20)
plt.show()

booster = xgb_classifier.get_booster()
importance = booster.get_score(importance_type='weight')
sorted_features = sorted(importance.items())
features, importances = zip(*sorted_features)
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances, align='center')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks([])
plt.show()

sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))






# 在LSUN的20类上训练一个xgboost



with open('clip_progan_train_multicls_real_features.pkl', 'rb') as file:
    loaded_train_dict = pickle.load(file)


train_features = loaded_train_dict['ProGAN_train_real'][0]
train_labels = loaded_train_dict['ProGAN_train_real'][1]

xgb_classifier = xgb.XGBClassifier(objective='multi:softprob', num_class=20, use_label_encoder=False, verbose_eval=True)

xgb_classifier.fit(train_features, train_labels)



with open('clip_progan_train_multicls_fake_features.pkl', 'rb') as file:
    loaded_test_dict = pickle.load(file)


test_features = loaded_test_dict['ProGAN_train_fake'][0]
test_labels = loaded_test_dict['ProGAN_train_fake'][1]

y_pred = xgb_classifier.predict_proba(test_features)







