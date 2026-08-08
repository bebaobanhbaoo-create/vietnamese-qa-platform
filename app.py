from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Vietnamese QA Platform'

@app.route('/health')
def health():
    return {'status': 'ok'}

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    return jsonify({'answer': 'Processing...'})
