from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend requests

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get("message", "")
    
    # Dummy response for now
    reply = f"你說的是：『{message}』，請問你還有其他關於租屋的問題嗎？"
    
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run()
