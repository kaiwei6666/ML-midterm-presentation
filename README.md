# 機器學習期中報告

> 提醒：只要去調整 `Group Activity/` 裡 `.py` 檔案裡的參數就好。

## 2026/5/5 新增內容

### 資料夾

- `Group Activity/`

### 檔案

- `Group Activity/ensemble_learning_evaluation.py`
- `Group Activity/ensemble_learning_results.csv`
- `Group Activity/hyperparameter_tuning_evaluation.py`
- `Group Activity/hyperparameter_tuning_results.csv`
- `Group Activity/pca_lda_evaluation.py`
- `Group Activity/pca_lda_results.csv`

### 增加 AUC

- `music_valence_classification.py`
- `valence_classification_metrics.csv`
- `valence_classification_report.md`

## 內容說明

- `ensemble_learning_evaluation.py`：比較不同集成學習模型
- `hyperparameter_tuning_evaluation.py`：測試超參數調整結果
- `pca_lda_evaluation.py`：測試 PCA 和 LDA 降維結果
- `.csv` 檔案：存放各個實驗的結果
- `music_valence_classification.py`：加入 AUC 計算

## 邱Output

### Baseline

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7696 | 0.7740 | 0.7678 | 0.7678 | 0.8308 |

### 1. PCA / LDA

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
| --- | --- | --- | --- | --- | --- |
| LDA (1) + Logistic Regression | 0.7696 | 0.7740 | 0.7678 | 0.7678 | 0.1659 |

### 2. Hyperparameter Tuning

| Model | Best Parameters | Accuracy | Precision | Recall | F1 Score | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| SVM | C = 1, gamma = scale, kernel = rbf, probability = true | 0.7696 | 0.7756 | 0.7676 | 0.7673 | 0.8001 |

### 3. Ensemble Learning

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
| --- | --- | --- | --- | --- | --- |
| StackingClassifier | 0.7592 | 0.7632 | 0.7574 | 0.7572 | 0.8218 |
