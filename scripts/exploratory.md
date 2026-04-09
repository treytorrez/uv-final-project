---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: final-proj
    language: python
    name: final-proj
---

```python
import pandas as pd
import polars as pl
import seaborn as sb
import lets_plot as lp
import sklearn
from sklearn.preprocessing import LabelEncoder
import numpy as np
pl.Config.set_tbl_cols(50)  # show up to 50 columns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score,
                             precision_recall_curve, auc, f1_score, make_scorer)
import amd_xgboost as xgb
from xgboost import XGBClassifier

```

```python
DATA_DIR = "../data/"
TRAIN_FRAC = .7
houses = [
 "csh101", 
 "csh102", 
 "csh103", 
 "csh104", 
 "csh105", 
 "csh106", 
 "csh107", 
 "csh108", 
 "csh109", 
 "csh110", 
 "csh111", 
 "csh112", 
 "csh113", 
 "csh114", 
 "csh115", 
 "csh116", 
 "csh117", 
 "csh118", 
 "csh119", 
 "csh120", 
 "csh121", 
 "csh122", 
 "csh123", 
 "csh124", 
 "csh125", 
 "csh126", 
 "csh127", 
 "csh128", 
 "csh129", 
 "csh130", 
]
print("Hello World!")

```


```python
df_list = [ pl.scan_csv( f"{DATA_DIR}raw/{house_name}/{house_name}.ann.features.csv") for house_name in houses[:30] ]
```

```python
#df_all = pl.concat(pl.collect_all(df_list), how='diagonal')
df_sampled = [df.collect().sample(fraction=.5) for df in df_list]
df_sampled = pl.concat(
    df_sampled,
    how='diagonal'
)
le = LabelEncoder()
```

```python
X = df_sampled.drop('activity')
y = le.fit_transform(df_sampled['activity'])


X_train, X_test, y_train, y_test = train_test_split(df_sampled.drop('activity'), le.fit_transform(df_sampled['activity']), test_size=0.33, random_state=42)
```

```python
# Train the model
model = xgboost.XGBClassifier( random_state=42, verbose=2)
model.fit(X_train, y_train)
```

```python
# predictions
from sklearn.metrics import f1_score


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # shape (n_samples, n_classes)

# basic metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred, average='weighted'))
#print(classification_report(y_test, y_pred, target_names=le.classes_, labels=))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ROC AUC (multiclass: one-vs-rest)
if y_proba.shape[1] == 2:
    roc = roc_auc_score(y_test, y_proba[:,1])
else:
    roc = roc_auc_score(y_test, y_proba, multi_class='ovr')
print("ROC AUC:", roc)

# Precision-Recall AUC for positive class (example for class 1)
precision, recall, _ = precision_recall_curve(y_test == 1, y_proba[:, 1])
pr_auc = auc(recall, precision)
print("PR AUC for class 1:", pr_auc)
```

```python

```
