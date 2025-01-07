from flask import Flask, request, jsonify, render_template
import joblib

# Initialize Flask app
app = Flask(__name__, template_folder='.')

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Serves the main HTML page

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json  # Get JSON data from the request
    review = data.get('review', '')  # Extract the 'review' field

    if not review:
        return jsonify({'error': 'No review provided'}), 400

    # Transform the review using the vectorizer
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

    return jsonify({'sentiment': sentiment})  # Return sentiment as JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
