from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('neural_network_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.form.to_dict()

        # Extract the features and convert to the appropriate data types
        account_length = int(data.get('account-length', 0))
        international_plan = int(data.get('international-plan', ''))
        voice_mail_plan = int(data.get('voice-mail-plan', ''))
        number_vmail_messages = int(data.get('number-vmail-messages', 0))
        total_day_minutes = float(data.get('total-day-minutes', 0.0))
        total_day_calls = int(data.get('total-day-calls', 0))
        total_day_charge = float(data.get('total-day-charge', 0.0))
        total_eve_minutes = float(data.get('total-eve-minutes', 0.0))
        total_eve_charge = float(data.get('total-eve-charge', 0.0))
        total_night_minutes = float(data.get('total-night-minutes', 0.0))
        total_night_charge = float(data.get('total-night-charge', 0.0))
        total_intl_minutes = float(data.get('total-intl-minutes', 0.0))
        total_intl_calls = int(data.get('total-intl-calls', 0))
        total_intl_charge = float(data.get('total-intl-charge', 0.0))
        customer_service_calls = int(data.get('customer-service-calls', 0))

        # Perform any necessary preprocessing on the features
        # ...

        # Create a feature vector for prediction
        features = np.array([
            account_length, international_plan, voice_mail_plan, number_vmail_messages,
            total_day_minutes, total_day_calls, total_day_charge,
            total_eve_minutes, total_eve_charge,
            total_night_minutes, total_night_charge,
            total_intl_minutes, total_intl_calls, total_intl_charge,
            customer_service_calls
        ], dtype=np.float32).reshape(1, -1)

        # Make the prediction using the model
        prediction_probabilities = model.predict(features)
        prediction = np.argmax(prediction_probabilities)

        # Prepare the prediction result
        result = 'Churn' if prediction == 1 else 'Not Churn'

        
        return jsonify({'prediction': result})

    except KeyError:
        error_message = 'One or more required fields are missing in the form data.'
        return jsonify({'error': error_message}), 400

if __name__ == '__main__':
    app.run()
