import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Gráficos adicionales
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')

"""
print(cancer.head())
print(cancer.info())
print(cancer.describe())
"""

# define target (y) and features (X)
y= cancer['diagnosis']
X= cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

"""
# check for missing values
print(x.isnull().sum())
print(y.isnull().sum())
# check for duplicates
print(x.duplicated().sum())
print(y.duplicated().sum())
"""

# train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# select model
model = LogisticRegression(max_iter=5000)

# train or fit model
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Show evaluation metrics
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nPrecisión:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))



#  Visual confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# ROC curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='M')
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Importance of characteristics (model coefficients)
plt.figure(figsize=(12, 8))
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Importancia de Características (Coeficientes de Regresión Logística)')
plt.show()

# Distribution of predictions
plt.figure(figsize=(8, 6))
sns.countplot(x=y_pred)
plt.title('Distribución de Predicciones')
plt.xlabel('Diagnóstico Predicho')
plt.ylabel('Cantidad')
plt.show()

# Further exploratory analysis
plt.figure(figsize=(12, 8))
sns.countplot(x=y)
plt.title('Distribución de Diagnósticos')
plt.show()
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title('Matriz de Correlación de Características')
plt.show()
