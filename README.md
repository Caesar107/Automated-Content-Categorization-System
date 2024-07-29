# Automated Content Categorization System

This repository contains the source code and documentation for our project, which aims to develop an intelligent system for automated content categorization using advanced machine learning algorithms.

## Overview

The objective of this project is to create a robust and accurate system capable of categorizing content automatically. By leveraging various machine learning techniques, we aim to achieve high accuracy and efficiency in content classification.

## Features

- **Data Collection**: Aggregates data from multiple sources, including online databases and APIs.
- **Data Preprocessing**: Cleans, normalizes, and extracts features from the raw data to prepare it for model training.
- **Model Selection**: Evaluates different machine learning models, such as decision trees, support vector machines, and neural networks, to find the most effective one.
- **Model Training**: Splits the data into training and testing sets and uses cross-validation to optimize model parameters.
- **Model Evaluation**: Assesses the performance of the model using metrics like accuracy, precision, recall, and F1-score.
- **Deployment**: Deploys the final model through a web-based interface, enabling users to input data and receive automated categorization results.

## Methodology

1. **Data Collection**: Gathered data from diverse sources to ensure a comprehensive dataset.
2. **Data Preprocessing**: Cleaned and normalized the data, extracting relevant features to enhance model performance.
3. **Model Selection**: Tested various models to identify the most effective one for our specific use case.
4. **Training**: Split the data into training and testing sets and used cross-validation to fine-tune model parameters.
5. **Evaluation**: Assessed the models using standard metrics and selected the best-performing model.
6. **Deployment**: Developed a user-friendly web interface for easy access to the categorization system.

## Code Description

The core code for this project is implemented in the `train_v2.py` file. Below is a brief overview of the main functionalities and processes:

### Main Functions

- **Data Loading**: The dataset is loaded from a CSV file using Pandas.
- **Data Preprocessing**: Text data is transformed into TF-IDF feature vectors to capture the importance of terms.
- **Model Training**: A Support Vector Machine (SVM) with a linear kernel is trained on the preprocessed data.
- **Model Evaluation**: The trained model is evaluated using accuracy, precision, recall, and F1-score metrics.

### Code Highlights

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('data.csv')

# Preprocess the data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
```

This code provides a solid foundation for automated content categorization, leveraging machine learning to deliver accurate and reliable results.

## Results

The selected model demonstrated high accuracy and robustness across various scenarios, making it suitable for real-world applications.

## Future Work

- **Enhancing Model Accuracy**: Continuously improving the model to achieve even higher accuracy.
- **Expanding Capabilities**: Adding more features to the system to broaden its applicability.

## Contributors

- Menglin Zou

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

