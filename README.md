# Handwritten Digit Classification with PCA and Ensemble Models

This project demonstrates how to classify handwritten digits using the `digits` dataset from `scikit-learn`. The workflow includes data preprocessing, dimensionality reduction using PCA, model training with GridSearchCV, and ensemble learning with a Voting Classifier.

## 📁 Dataset
- **Source:** `sklearn.datasets.load_digits`
- **Description:** The dataset contains 8x8 pixel grayscale images of handwritten digits (0-9).

## ⚙️ Technologies Used
- Python 3.x
- scikit-learn
- NumPy
- Matplotlib

## 📊 Project Workflow

1. **Data Loading & Visualization:**
   - Display the first 10 digit images.

2. **Preprocessing:**
   - Standardize the feature set using `StandardScaler`.

3. **Dimensionality Reduction:**
   - Apply PCA to retain 95% of variance.
   - Also reduce to 2D for visualization purposes.

4. **Model Training with Grid Search:**
   - **Support Vector Machine (SVM)**
   - **Random Forest Classifier (RF)**
   - **K-Nearest Neighbors (KNN)**
   - Hyperparameter tuning via `GridSearchCV`.

5. **Ensemble Learning:**
   - Combine the best models using a `VotingClassifier`.

6. **Evaluation:**
   - Display the confusion matrix.
   - Print best parameters and final accuracy.

### Run the project

```bash
git clone https://github.com/ozgeebasknn/proje_adi.git
cd proje_adi
pip install -r requirements.txt
python proje_adi.py