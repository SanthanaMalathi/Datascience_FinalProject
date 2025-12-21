"""Credit Card Fraud Detection using Machine Learning"""
# 1.Import Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import  Pipeline
from sklearn.metrics import (classification_report,precision_score,recall_score
                            ,roc_auc_score,roc_curve,precision_recall_curve,average_precision_score,
                             confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

#2. Load the Dataset
df=pd.read_csv("credit_card.csv")
print(df)

#3.Check the basic information about the dataset
df_columns=df.columns
print("Column Details:\n",df_columns)
print("----------------------------")
print("Data types of Each column:\n",df.dtypes)
print("----------------------------")
rows,columns=df.shape
print("No of rows:",rows)
print("No of columns:",columns)
print("----------------------------")
print("No of elements:",df.size)
print("----------------------------")
print("Indexes:",df.index)
print("----------------------------")
print("DataFrame Info:")
print(df.info())
print("----------------------------")
print("\nDescriptive statistics for numerical columns:")
print(df.describe())
print("----------------------------")
print("Total missing values")
print(df.isna().sum())
print("----------------------------")
print("Checking for duplicate rows")
print(df[df.duplicated()])
df.drop_duplicates(inplace=True)
print("Duplicate rows removed successfully")
print("----------------------------")

# 4.Target Variable Analysis
print("Count of 'is_fraud' values:")
print(df['Class'].value_counts())
print("\nPercentage of 'is_fraud' values:")
print(df['Class'].value_counts(normalize=True) * 100)
"""visualization of target"""
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Distribution of Target Variable (is_fraud)')
plt.xlabel('Is Fraud')
plt.ylabel('Number of Transactions')
plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'])
plt.show()
# Transaction Amount Distribution

plt.figure(figsize=(12,4))

# 5. Feature Analysis & Engineering
# Transaction Amount
plt.subplot(1,2,1)
sns.histplot(df['Amount'], bins=80, log_scale=True)
plt.title("Amount Distribution (Log Scale)")

plt.subplot(1,2,2)
sns.boxplot(x=df['Amount'])
plt.title("Amount Boxplot")

plt.tight_layout()
plt.show()

# Log Transformation (important)
df['Amount_log'] = np.log1p(df['Amount'])
df.drop('Amount', axis=1, inplace=True)
# Time Feature Engineering
df['Hour'] = (df['Time'] // 3600) % 24
df.drop('Time', axis=1, inplace=True)

#6.Correlation Heatmap
plt.figure(figsize=(20,12))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.1)
plt.title("Correlation Heatmap")
plt.show()
print("----------------------------")

"""Data Splitting"""
# Separate features (X) and target variable (y)
X = df.drop('Class', axis=1)
y = df['Class']

#7.Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Dataset split into training and testing sets:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nClass distribution in original data:")
print(y.value_counts(normalize=True))
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))
print("\nClass distribution in testing set:")
print(y_test.value_counts(normalize=True))

# Scale data for ANOVA visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Fit ANOVA to get F-scores
anova_selector = SelectKBest(score_func=f_classif, k='all')
anova_selector.fit(X_scaled, y_train)

anova_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'F_Score': anova_selector.scores_,
    'P_Value': anova_selector.pvalues_
}).sort_values(by='F_Score', ascending=False)

# Show top 10 features
print(anova_scores.head(10))

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='F_Score', y='Feature', data=anova_scores.head(15), palette='viridis')
plt.title("Top 15 Features by ANOVA F-Score")
plt.xlabel("F-Score")
plt.ylabel("Feature")
plt.show()

# 8. Baseline Models (with SMOTE applied correctly)
# Logistic Regression Pipeline
lr_pipeline = Pipeline([('smote', SMOTE(sampling_strategy=0.2, random_state=42)),
                        ('anova', SelectKBest(score_func=f_classif, k=15)),
                        ('scaler', StandardScaler()),
                        ('model', LogisticRegression(max_iter=1000))])

# Random Forest Pipeline
rf_pipeline = Pipeline([ ('smote', SMOTE(sampling_strategy=0.2, random_state=42)),
                         ('anova', SelectKBest(score_func=f_classif, k=15)),
                         ('model', RandomForestClassifier(random_state=42))])

# XGBoost Pipeline
xgb_pipeline = Pipeline([('smote', SMOTE(sampling_strategy=0.2, random_state=42)),
                         ('anova', SelectKBest(score_func=f_classif, k=15)),
                         ('model', XGBClassifier(eval_metric='logloss',n_jobs=-1,random_state=42,
                            use_label_encoder=False))])

# Gradient Boosting pipeline
gb_pipeline = Pipeline([ ('smote', SMOTE(sampling_strategy=0.2, random_state=42)),
                         ('anova', SelectKBest(score_func=f_classif, k=15)),
                         ('model', GradientBoostingClassifier(random_state=42))])


# 9. Model Evaluation Function
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(f"\n===== {name} =====")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC:", average_precision_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # Plot PR curve immediately after confusion matrix
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_prob

# 10. Evaluate Baseline Models + PR Curve
lr_probs  = evaluate_model("Logistic Regression", lr_pipeline)
rf_probs  = evaluate_model("Random Forest", rf_pipeline)
xgb_probs = evaluate_model("XGBoost", xgb_pipeline)
gb_probs = evaluate_model("Gradient Boosting",gb_pipeline)

# 11. ROC Curve Comparison
plt.figure(figsize=(7,6))

for name, probs in zip(
    ["Logistic Regression", "Random Forest", "XGBoost","Gradient Boosting"],
    [lr_probs, rf_probs, xgb_probs, gb_probs]
):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# 12. Hyperparameter Tuning
# Random Forest
rf_params = {"model__n_estimators": [50, 100],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__class_weight": ["balanced"]}

rf_search = RandomizedSearchCV(rf_pipeline, rf_params, n_iter=10,  scoring='f1',
                                cv=3,n_jobs=-1,random_state=42)
rf_search.fit(X_train, y_train)
# XGBoost
xgb_params = {"model__n_estimators": [50, 100],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1],
                "model__scale_pos_weight": [5, 10]}

xgb_search = RandomizedSearchCV(xgb_pipeline,xgb_params,n_iter=10,scoring='f1',
                                 cv=3,n_jobs=-1,random_state=42)

xgb_search.fit(X_train, y_train)

# 13. Final Model Comparison
final_models = {
    "Random Forest (Tuned)": rf_search.best_estimator_,
    "XGBoost (Tuned)": xgb_search.best_estimator_
}

results = []

for name, model in final_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    results.append({
        "Model": name,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob)
    })
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {name}:")
    print(cm)

metrics_df = pd.DataFrame(results)
print(metrics_df)

# 14. Metric Visualization
metrics_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(10,6))
sns.barplot(x="Model", y="Score", hue="Metric", data=metrics_long)
plt.title("Final Model Performance Comparison")
plt.xticks(rotation=20)
plt.show()

# Select Random Forest model explicitly (best performer)
best_model_name = "Random Forest (Tuned)"
best_model = final_models[best_model_name]

print(f"Selected Model for Deployment: {best_model_name}")

# Save model as pickle file
model_filename = "credit_card_fraud_random_forest.pkl"
joblib.dump(best_model, model_filename)

print(f"Model saved successfully as '{model_filename}'")
