# SMS Spam Detection

A Jupyter Notebook project for classifying SMS messages as **ham** or **spam** using a word-and-character TF-IDF feature pipeline with a linear Support Vector Machine (SVM).

## Overview

The notebook [`SMS_Spam_SVM.ipynb`](SMS_Spam_SVM.ipynb) demonstrates an accuracy-focused machine-learning workflow:

- Loads the SMS spam dataset from `spam.csv` or `SMS.csv`
- Accepts the common `v1`/`v2` dataset columns or `label`/`message` columns
- Removes missing and duplicate records
- Splits the data into stratified training and test sets
- Combines word-level and character-level TF-IDF features
- Tunes a `LinearSVC` model with cross-validation
- Reports test accuracy, a classification report, and a confusion matrix
- Displays misclassified messages for error analysis
- Predicts whether new messages are ham or spam

## Requirements

- Python 3
- Jupyter Notebook or Google Colab
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

The notebook includes a setup cell that installs the required Python packages:

```bash
pip install scikit-learn pandas matplotlib seaborn
```

## Dataset

Place one of the following files in the notebook's working directory:

- `spam.csv`
- `SMS.csv`

The expected dataset columns are either:

- `v1`: label and `v2`: message, as used by the common Kaggle/UCI SMS Spam Collection dataset
- `label`: label and `message`: message

Labels should be `ham` or `spam`.

## Usage

1. Open `SMS_Spam_SVM.ipynb` in Jupyter Notebook or Google Colab.
2. Add the dataset file to the working directory. In Google Colab, the notebook can prompt for an upload.
3. Run the notebook cells from top to bottom.
4. Review the cross-validation results, held-out test metrics, confusion matrix, and sample predictions.

The notebook defines a helper function for new messages:

```python
predict_sms("Congratulations! You have won a free iPhone. Click here now.")
```

## Model details

The classifier uses:

- Word TF-IDF features with unigrams and bigrams
- Character TF-IDF features with 3–5 character n-grams
- `LinearSVC` classification
- Grid search over SVM regularization and class-weight settings
- Five-fold stratified cross-validation

The final test accuracy is calculated when the notebook is executed. No fixed score is reported here because results depend on the dataset and runtime environment.

## License

No license has been specified for this repository yet.
