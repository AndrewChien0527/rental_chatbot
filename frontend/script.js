const chatBox = document.getElementById("chat-box");
const inputField = document.getElementById("user-input");

// Update with your actual backend URL on Render
const BACKEND_URL = "https://your-backend-url.onrender.com/chat";

function appendMessage(text, sender) {
  const message = document.createElement("div");
  message.classList.add("message", sender);
  message.textContent = text;
  chatBox.appendChild(message);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const userInput = inputField.value.trim();
  if (!userInput) return;

  appendMessage(userInput, "user");
  inputField.value = "";

  try {
    const res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userInput }),
    });
    const data = await res.json();
    appendMessage(data.reply, "bot");
  } catch (err) {
    appendMessage("⚠️ 無法連線到伺服器，請稍後再試。", "bot");
  }
}
