# Spam SMS / Email Classifier

A machine learning project that classifies text messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and TF-IDF feature extraction. Multiple machine learning classifiers are evaluated to identify the best-performing model.

## Project Overview

Spam messages are unwanted messages that may contain misleading information, advertisements, or fraudulent links. This project develops a text classification system that can automatically distinguish spam messages from legitimate messages.

The project includes data preprocessing, exploratory data analysis, model comparison, evaluation, model serialization, and example predictions.

## Objectives

- Build a machine learning model to classify SMS messages as spam or ham.
- Preprocess and transform text data using TF-IDF.
- Train and compare multiple machine learning classifiers.
- Evaluate performance using accuracy, precision, recall, and F1-score.
- Visualize model performance and the confusion matrix.
- Save the best-performing model for future predictions.

## Dataset

**Dataset:** [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The dataset contains SMS messages labeled as:
- `ham` — legitimate messages
- `spam` — unwanted messages

### Data Preprocessing

- Loaded the dataset using Pandas.
- Removed unnecessary columns.
- Renamed the target and message columns for clarity.
- Removed duplicate records.
- Converted the target labels into numerical values: ham = 0 and spam = 1.
- Split the dataset into training and testing sets using an 80:20 ratio with stratification.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Matplotlib and Seaborn
- Joblib
- XGBoost

## Machine Learning Models

The following 11 classifiers were compared:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Multinomial Naive Bayes
4. Decision Tree
5. K-Nearest Neighbors (KNN)
6. Random Forest
7. AdaBoost
8. Bagging Classifier
9. Extra Trees
10. Gradient Boosting
11. XGBoost

TF-IDF was used to convert text messages into numerical features for classification.

## Model Evaluation

The models were evaluated using accuracy, precision, recall, F1-score, and training time.

### Best Model: Support Vector Machine (SVM)

| Metric | Score |
|---|---:|
| Accuracy | 98.07% |
| Precision | 98.26% |
| Recall | 86.26% |
| F1-score | 91.87% |
| Training time | 6.52 seconds |

The SVM classifier achieved the highest F1-score among the evaluated models in this experiment and was saved as the final model.

**Important limitation:** Although the model achieved high accuracy and precision, its spam recall was 86.26%. This means some spam messages in the test set were classified as legitimate messages. Performance on new, real-world messages may differ.

These results are based on the project's current train-test split and are not a guarantee of future performance.

## Project Files

| File | Description |
|---|---|
| `spam.csv` | Dataset used for training and evaluation |
| `spam_classifier.ipynb` | Notebook containing preprocessing, training, and evaluation |
| `spam_classifier.pkl` | Serialized best-performing model and text-processing pipeline |
| `requirements.txt` | Python dependencies |
| `metrics.csv` | Evaluation metrics for the best model |
| `model_comparison.csv` | Performance comparison of all evaluated models |
| `example_predictions.csv` | Example predictions with labels and confidence information |
| `confusion_matrix.png` | Confusion matrix for model evaluation |
| `model_comparison.png` | Model performance comparison visualization |
| `class_distribution.png` | Distribution of spam and ham messages |
| `message_length_distribution.png` | Distribution of message lengths |

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/NishatLuba/AME_2_Spam-SMS_Email-Classifier.git
cd AME_2_Spam-SMS_Email-Classifier
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Load the saved model

The saved model contains the trained text-processing and classification pipeline.

```python
import joblib

model = joblib.load("spam_classifier.pkl")

message = ["Congratulations! You have won a free prize. Call now!"]

prediction = model.predict(message)[0]

if prediction == 1:
    print("Spam")
else:
    print("Ham (Not Spam)")
```

The model expects raw text messages as input. The example assumes the saved pipeline includes the fitted TF-IDF vectorizer.

**Note:** Only load serialized model files from sources you trust.

## Visualizations

The repository includes plots for:
- Class distribution
- Message length distribution
- Model performance comparison
- Confusion matrix

These visualizations help assess class balance, compare classifiers, and examine classification errors.

## Future Improvements

- Improve spam recall while maintaining strong precision.
- Tune hyperparameters using cross-validation.
- Evaluate additional text preprocessing and feature extraction techniques.
- Test the model on independent and more recent datasets.
- Develop a simple web interface for interactive predictions.

## Conclusion

This project demonstrates an end-to-end text classification workflow using NLP and machine learning. Eleven classifiers were evaluated, and SVM achieved the best F1-score in the current experiment. The final model was serialized for reuse, with evaluation reports and visualizations included to support reproducibility and analysis.

## Author

**Nishat Tasnim Luba**

- GitHub: [NishatLuba](https://github.com/NishatLuba)
- LinkedIn: [Nishat Tasnim Luba](https://www.linkedin.com/in/nishat-tasnim-luba/)
