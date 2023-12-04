# web_app.py
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Загрузка параметров модели
saved_weights, saved_bias = joblib.load('model_parameters.joblib')

# Заглушка для веб-приложения
class StubModel:
    def predict(self, features):
        z = np.dot(features, saved_weights) + saved_bias
        return 1 / (1 + np.exp(-z))

web_model = StubModel()

@app.route('/')
def index():
    return render_template('index_custom.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        num1 = request.form['num1']
        num2 = request.form['num2']
        print(f"Введенные числа: {num1}, {num2}")

        # Преобразование введенных текстов в числа
        features = np.array([float(num1), float(num2)])
        print(f"Преобразованные числа: {features}")

        # Проверка на корректность результата предсказания
        prediction = web_model.predict(features)
        print(f"Результат предсказания: {prediction}")

        if np.isnan(prediction) or not (0 <= prediction <= 1):
            raise ValueError("Модель вернула некорректный результат")

        # Определение класса на основе порогового значения 0.5
        predicted_class = 1 if prediction > 0.5 else 0

        return jsonify({'predicted_class': predicted_class})
    except ValueError as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
