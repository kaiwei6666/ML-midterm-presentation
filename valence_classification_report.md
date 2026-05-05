# Music Valence Classification Report

## Experiment Setup
- Dataset: `spotify-2023.csv`
- Samples: 953
- Parsing note: the source CSV has malformed quotes in some text columns, so the requested music features were extracted from the tail of each raw line to preserve valid records.
- Features: bpm, key, mode, danceability_%, energy_%, acousticness_%, instrumentalness_%, liveness_%, speechiness_%
- Label rule: `valence_% >= 51.0` -> `High Valence`, otherwise `Low Valence`
- Train/Test split: 762 / 191 (stratified, random_state=42)

## Class Distribution
| Class | Count |
| --- | --- |
| High Valence | 490 |
| Low Valence | 463 |

## Missing Values in Selected Features
| Feature | Missing Count |
| --- | --- |
| bpm | 0 |
| key | 0 |
| mode | 0 |
| danceability_% | 0 |
| energy_% | 0 |
| acousticness_% | 0 |
| instrumentalness_% | 0 |
| liveness_% | 0 |
| speechiness_% | 0 |

## Model Comparison
| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.7696 | 0.774 | 0.7678 | 0.7678 |
| Support Vector Machine (SVM) | 0.7696 | 0.7756 | 0.7676 | 0.7673 |
| Perceptron | 0.712 | 0.7145 | 0.7103 | 0.71 |
| K-Nearest Neighbors (KNN) | 0.6754 | 0.6752 | 0.6749 | 0.675 |
| Decision Tree | 0.6597 | 0.6594 | 0.6593 | 0.6593 |

## Confusion Matrices
### Decision Tree
|   | Predicted High Valence | Predicted Low Valence |
| --- | --- | --- |
| Actual High Valence | 66 | 32 |
| Actual Low Valence | 33 | 60 |

### Logistic Regression
|   | Predicted High Valence | Predicted Low Valence |
| --- | --- | --- |
| Actual High Valence | 82 | 16 |
| Actual Low Valence | 28 | 65 |

### Support Vector Machine (SVM)
|   | Predicted High Valence | Predicted Low Valence |
| --- | --- | --- |
| Actual High Valence | 83 | 15 |
| Actual Low Valence | 29 | 64 |

### Perceptron
|   | Predicted High Valence | Predicted Low Valence |
| --- | --- | --- |
| Actual High Valence | 76 | 22 |
| Actual Low Valence | 33 | 60 |

### K-Nearest Neighbors (KNN)
|   | Predicted High Valence | Predicted Low Valence |
| --- | --- | --- |
| Actual High Valence | 68 | 30 |
| Actual Low Valence | 32 | 61 |

## Conclusion
- Best model: `Logistic Regression` with Accuracy `0.7696` and Macro F1 `0.7678`.
- These audio features show moderate predictive power for valence, but they are not strong enough for highly reliable emotion classification on their own.
