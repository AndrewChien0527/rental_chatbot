from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

# Root route to prevent 404 on Render
@app.route("/")
def home():
    return jsonify({"message": "Rental chatbot backend is working!"})

# Load embedder & FAISS index
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
with open("vectorstore/faiss_index.pkl", "rb") as f:
    index, docs = pickle.load(f)

# Load small language model (SLM)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True).half().cpu()
model.eval()


def classify_input(text):
    keywords_q2 = ["地點", "地址", "坪數", "租金", "格局", "樓層"]
    if sum(k in text for k in keywords_q2) >= 2:
        return "Q2"
    elif any(word in text for word in ["房東", "租約", "押金", "修繕", "漲價", "合約"]):
        return "Q1"
    elif any(word in text for word in ["Thanks", "OK", "ok", "Ok", "謝謝", "感恩"]):
        return "greeting"
    else:
        return "other"


def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def handle_general_question(query):
    vec = embed_model.encode([query])
    vec = vec.reshape(1, -1).astype('float32')
    _, idxs = index.search(vec, k=3)
    context = "\n".join([docs[i] for i in idxs[0]])
    prompt = f"""根據以下租屋知識回答問題：\n{context}\n\n問題：{query}\n回答："""
    return generate_response(prompt)


def handle_rental_post(text):
    required_slots = {
        "地點": r"地點[:：]\s*(.*)",
        "地址": r"地址[:：]\s*(.*)",
        "格局": r"格局[:：]\s*(.*)",
        "坪數": r"坪數[:：]\s*(.*)",
        "樓層": r"樓層[:：]\s*(.*)",
        "租金": r"租金[:：]\s*(.*)",
    }
    found = {}
    for slot, pattern in required_slots.items():
        match = re.search(pattern, text)
        if match:
            found[slot] = match.group(1).strip()

    missing = [k for k in required_slots if k not in found]
    filled = [f"✅ {k}：{v}" for k, v in found.items()]
    missing_msg = [f"⚠️ 缺少「{m}」，可能導致租屋風險，建議向房東確認" for m in missing]

    summary = "\n".join(filled + missing_msg)
    prompt = f"""根據以下貼文資料，請說明租屋風險與建議：\n{text}\n\n目前擷取資訊如下：\n{summary}\n\n請總結風險與建議："""
    return generate_response(prompt)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    intent = classify_input(user_message)

    if intent == "Q1":
        answer = handle_general_question(user_message)
    elif intent == "Q2":
        answer = handle_rental_post(user_message)
    else:
        answer = "請提供與租屋相關的問題或貼文內容～"

    return jsonify({"response": answer})
