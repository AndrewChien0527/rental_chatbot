import torch
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load embedder & FAISS index
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
with open("vectorstore/faiss_index.pkl", "rb") as f:
    index, docs = pickle.load(f)

# Load SLM (可替換為Qwen1.5/ChatGLM3/baichuan2)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True).half().cpu()
model.eval()

def classify_input(text):
    """分類為 Q1 (一般問題)、Q2 (FB貼文)、other"""
    keywords_q2 = ["地點", "地址", "坪數", "租金", "格局", "樓層"]
    if sum(k in text for k in keywords_q2) >= 2:
        return "Q2"
    elif any(word in text for word in ["房東", "租約", "押金", "修繕", "漲價", "合約"]):
        return "Q1"
    elif any(word in text for word in ["Thanks","OK","ok","Ok","謝謝","感恩"]):
        return "greeting"
    else:
        return "other"
    

def classify_user_input(user_input: str, model):
    prompt = f"""
請根據使用者的輸入，判斷這是屬於哪一類租屋問題，請只回覆分類代碼（不解釋理由）：

1. legal_issue：法律爭議、押金、提前終止、房東違約、漲租等  
2. listing_analysis：貼文解析，例如 FB 文章、租屋廣告等資訊分析  
3. greeting：打招呼、寒暄、感謝、結尾等  
4. rental_advice：非法律性租屋建議、地區選擇、生活機能比較等  
5. contract_clause：合約條款說明、租約句子解析  
6. info_checker：希望系統幫忙檢查缺漏資訊  
7. escalation：需要轉介、超出系統能力、詢問是否該找專家

範例：
使用者：房東說押金不退，這合理嗎？
分類：legal_issue

使用者：你好啊～
分類：greeting

使用者：我覺得這房子看起來還不錯，你覺得呢？（附上一篇租屋文）
分類：listing_analysis

使用者：你可以幫我看看我描述的內容還缺什麼資料嗎？
分類：info_checker

使用者：我該選新店還是台大附近租房？
分類：rental_advice

使用者：以下是合約上的一段話，你幫我解釋：
分類：contract_clause

使用者：這樣我是不是該找律師？
分類：escalation

使用者：{user_input}
分類：
"""
    output = model.generate(prompt)
    return output.strip()


def classify_with_prompt(sl_model, text):
    prompt = f"""
你是一個租屋法律聊天機器人，負責判斷使用者輸入屬於哪一類。分類如下：

- Q1：詢問租屋法律相關問題，例如：房東責任、押金爭議、修繕、租約糾紛等。
- Q2：提供租屋貼文格式的內容，例如：地點、坪數、租金、格局、樓層等資訊。
- response：補充缺漏資訊或一般感謝、回復用語
- else: 無法分類、或與租屋無關的內容，例如：「謝謝」、「OK」、「你好」。

請你只回覆其中一個字串：Q1、Q2、response、else。

使用者輸入：
{text}

你的回答是：
"""
    response = sl_model.generate(prompt).strip()
    return response if response in ["Q1", "Q2", "response","else"] else "other"


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
