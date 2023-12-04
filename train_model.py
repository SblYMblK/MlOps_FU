# train_model.py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Генерация синтетического датасета
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

# Разделение на тренировочную и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели (логистическая регрессия)
model = LogisticRegression()
model.fit(X_train, y_train)

# Оценка качества модели
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy}")

# Сохранение параметров модели
saved_weights = model.coef_[0]
saved_bias = model.intercept_[0]

# Сохранение модели в файл
joblib.dump((saved_weights, saved_bias), 'model_parameters.joblib')
