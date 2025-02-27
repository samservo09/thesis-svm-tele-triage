from flask import Flask, request, jsonify, session
from flask_session import Session  # pip install Flask-Session
import pickle

app = Flask(__name__)

# --- Session config for demonstration purposes (using server-side session) ---
app.config["SECRET_KEY"] = "YOUR_SECRET_KEY"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load your model if needed (for classification after questions)
# with open('svm_model.pkl', 'rb') as f:
#     model_data = pickle.load(f)
#     model = model_data["random_search"]
#     tfidf_vectorizer = model_data["tfidf_vectorizer"]

# The Columbia Protocol questions (simplified for demonstration).
QUESTIONS = [
    "1) Have you wished you were dead or wished you could go to sleep and not wake up?",
    "2) Have you actually had any thoughts about killing yourself?",
    "3) Have you been thinking about how you might do this?",
    "4) Have you had these thoughts and had some intention of acting on them?",
    "5) Have you started to work out or worked out the details of how to kill yourself? Did you intend to carry out the plan?",
    "6) Have you done anything, started to do anything, or prepared to do anything to end your life?"
]

HIGH_RISK_QUESTIONS = [2, 3, 4, 5, 6]  
# If user answers "YES" to question #2-#6, it's high risk => immediate intervention

@app.route('/')
def index():
    return "Hello! This is a demonstration of a multi-step suicide risk assessment chatbot."

@app.route('/start_assessment', methods=['GET'])
def start_assessment():
    """Initialize the conversation and ask the first question."""
    session['current_question'] = 0
    session['answers'] = []
    
    return jsonify({
        "message": "Starting the Columbia Protocol. " + QUESTIONS[0]
    })

@app.route('/answer', methods=['POST'])
def answer_question():
    """User sends an answer to the current question. 
       We then move to the next question or provide classification/intervention if needed."""
    data = request.get_json()
    user_answer = data.get('answer', '').strip().lower()

    current_q_index = session.get('current_question', 0)
    answers = session.get('answers', [])
    
    # Store the answer
    answers.append(user_answer)
    session['answers'] = answers
    
    # Check if the user answered "yes" to a high-risk question
    # (Note: question indices start at 0, but "high-risk" is 2..6 => we adjust)
    if user_answer.startswith("yes") and (current_q_index + 1) in HIGH_RISK_QUESTIONS:
        # Immediate intervention
        return jsonify({
            "message": (
                "Thank you for sharing. Based on your answer, I'm concerned about your safety. "
                "Please consider calling 911 (or local emergency) or 988 (in the US). "
                "Stay with someone you trust or reach out to a mental health professional immediately."
            ),
            "end": True
        })
    
    # If we've not reached the last question, move on
    if current_q_index < len(QUESTIONS) - 1:
        current_q_index += 1
        session['current_question'] = current_q_index
        
        # If the user answered "no" to Q2, skip to Q6 (like the protocol says)
        # (Q2 is index 1 => if "no", skip Q3-Q5 => jump to Q6 (index 5))
        if current_q_index == 2 and answers[1].startswith("no"):
            session['current_question'] = 5  # jump to question 6
            return jsonify({
                "message": QUESTIONS[5]
            })
        
        # Otherwise, ask the next question
        return jsonify({
            "message": QUESTIONS[current_q_index]
        })
    else:
        # We have all answers
        # Here you could do a final classification or summary
        # For demonstration, let's do a simple rule-based classification
        risk_level = classify_risk(answers)
        
        # Optionally run your SVM classification if you have textual input
        # user_text = " ".join(answers)  # if your model is text-based
        # X_input_tfidf = tfidf_vectorizer.transform([user_text])
        # predicted_class = model.predict(X_input_tfidf)[0]
        
        # Return final response
        return jsonify({
            "message": f"Assessment complete. Based on your answers, your risk level is: {risk_level}. "
                       "If you are in crisis, please reach out for help immediately.",
            "end": True
        })

def classify_risk(answers):
    """
    A dummy classification function that checks how many 'yes' answers we have.
    In a real scenario, you'd have a more nuanced approach or an actual ML model.
    """
    yes_count = sum(ans.startswith("yes") for ans in answers)
    if yes_count >= 3:
        return "HIGH RISK"
    elif yes_count == 2:
        return "MODERATE RISK"
    else:
        return "LOW RISK"

if __name__ == '__main__':
    app.run(debug=True)
