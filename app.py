from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load("svm-roberta_model.pkl")

# Define CSSR-S Questions (structured sequence)
cssrs_questions = [
    "Have you wished you were dead or wished you could go to sleep and not wake up?",
    "Have you actually had any thoughts of killing yourself?",
    "Have you been thinking about how you might do this?",
    "Have you had these thoughts and had some intention of acting on them?",
    "Have you started to work out or worked out the details of how to end your life?",
    "Have you done anything, started to do anything, or prepared to do anything to end your life?"
]

user_responses = {}

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    
    session_id = data['session']
    intent = data['queryResult']['intent']['displayName']
    user_input = data['queryResult']['queryText']

    # Track user responses
    if session_id not in user_responses:
        user_responses[session_id] = []
    
    user_responses[session_id].append(user_input)

    # If all CSSR-S questions are answered, classify
    if len(user_responses[session_id]) >= len(cssrs_questions):
        # Convert responses into a NumPy array for classification
        user_features = np.array(user_responses[session_id]).reshape(1, -1)
        prediction = model.predict(user_features)[0]

        # Clear session responses after classification
        del user_responses[session_id]

        return jsonify({
            "fulfillmentText": f"Based on your responses, your classification is: {prediction}"
        })
    
    # Ask next question
    next_question = cssrs_questions[len(user_responses[session_id])]
    return jsonify({"fulfillmentText": next_question})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
