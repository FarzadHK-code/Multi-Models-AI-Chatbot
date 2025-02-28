import gradio as gr
import pathlib
import os
import sqlite3
import psutil
import subprocess
import threading
import time
import requests
import ollama
import base64
import json
import re

# Initialize SQLite Database
def initialize_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_prompt TEXT,
            model_response TEXT,
            model_used TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

initialize_db()

# Available AI Models
models = {
    "General Chat": "deepseek-r1:latest",
    "Coding": "deepseek-coder:latest",
    "Large Context": "codellama:70b",
    "Advanced Reasoning": "phi4:latest",
    "General Knowledge": "gemma:latest",
    "Lightweight Model": "mistral:latest",
    "Image Generation": "stability-api"
}

# Get all model names as a string for UI display
all_models_str = ", ".join(models.values())

# Function to determine the best model
def select_model(prompt):
    prompt_lower = prompt.lower()

    # ✅ Use "stability-api" instead of "stablediffusion"
    if any(keyword in prompt_lower for keyword in ["generate an image", "draw", "illustrate", "paint", "visualize"]):
        model = "stability-api"  # ✅ Fix: Stability AI API instead of Ollama's Stable Diffusion

    # ✅ Coding Model
    elif any(keyword in prompt_lower for keyword in ["code", "script", "function", "debug", "fix", "programming", "develop"]):
        model = models["Coding"]

    # ✅ Large Context Model (Summarization, Documents)
    elif any(keyword in prompt_lower for keyword in ["summarize", "document", "pdf", "long text", "large context", "explain this article"]):
        model = models["Large Context"]

    # ✅ Advanced Reasoning Model (Math, Physics, Scientific Theories)
    elif any(keyword in prompt_lower for keyword in ["math", "logic", "reasoning", "complex problem", "solve", "equation", "riddle", "brain teaser",
                                                     "explain the theory", "scientific principle", "mechanics", "relativity", "quantum", "einstein", "physics"]):
        model = models["Advanced Reasoning"]

    # ✅ Quick Facts (Use Mistral for simple Q&A)
    elif any(keyword in prompt_lower for keyword in ["quick fact", "trivia", "short answer", "fastest", "biggest", "smallest", "short response"]):
        model = models["Lightweight Model"]

    # ✅ General Knowledge (History, Science, Geography, etc.)
    elif any(keyword in prompt_lower for keyword in ["history", "science", "geography", "fact", "knowledge", "who", "what", "where", "when"]):
        model = models["General Knowledge"]

    # ✅ Default to General Chat if no conditions match
    else:
        model = models["General Chat"]

    # 🔍 Debugging - Log Selected Model
    print(f"✅ Model Selected: {model} for prompt: '{prompt}'")

    return model

    # ✅ Model Selection Based on Keywords
    if any(keyword in prompt_lower for keyword in image_keywords):
        model = models["Image Generation"]
    elif any(keyword in prompt_lower for keyword in coding_keywords):
        model = models["Coding"]
    elif any(keyword in prompt_lower for keyword in general_chat_keywords):
        model = models["General Chat"]
    elif any(keyword in prompt_lower for keyword in large_context_keywords):
        model = models["Large Context"]
    elif any(keyword in prompt_lower for keyword in advanced_reasoning_keywords):
        model = models["Advanced Reasoning"]
    elif any(keyword in prompt_lower for keyword in general_knowledge_keywords):
        model = models["General Knowledge"]
    elif any(keyword in prompt_lower for keyword in lightweight_keywords):
        model = models["Lightweight Model"]
    else:
        model = models["Lightweight Model"]  # Fallback to lightweight model

    # 🔍 Debugging - Log Selected Model
    print(f"✅ Model Selected: {model} for prompt: '{prompt}'")

    return model

# Function to generate images using Stability.ai API
def generate_image(prompt):
    """Generates an image using Stability AI API and saves it locally."""

    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        return "Error: Stability API key is missing."

    print("API Key:", api_key)  # Debugging

    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*"  # ✅ Fix: Ensure API expects an image
    }
    
    files = {
        "prompt": (None, prompt),
        "width": (None, "512"),
        "height": (None, "512"),
        "output_format": (None, "png")
    }

    response = requests.post(url, files=files, headers=headers)

    print("Response Status Code:", response.status_code)
    print("Response Headers:", response.headers)

    content_type = response.headers.get("Content-Type", "")

    # ✅ Check if the response contains an image
    if response.status_code == 200 and "image" in content_type:
        timestamp = int(time.time())  # Unique filename
        image_path = f"generated_image_{timestamp}.png"

        with open(image_path, "wb") as f:
            f.write(response.content)

        if os.path.exists(image_path):
            print(f"✅ Image successfully saved as {image_path}")
            return image_path  # ✅ Return valid image path
        else:
            return "Error: Failed to save the image."

    else:
        # ✅ Handle JSON error response properly
        try:
            error_details = response.json()
        except:
            error_details = response.text
        print("❌ Error Response:", error_details)
        return f"Error: {error_details}"

# Function to load previous chats
def load_chat_history():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_prompt FROM chats ORDER BY timestamp DESC")
    chats = cursor.fetchall()
    conn.close()
    
    chat_list = [f"Chat {chat_id}: {prompt[:30]}..." for chat_id, prompt in chats] if chats else ["No previous chats"]
    
    print("🔍 Loaded chat history:", chat_list)  # ✅ Debug print
    
    return chat_list

# Function to monitor CPU, RAM
def get_system_resources():
    cpu_usage = psutil.cpu_percent(interval=0.5)
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = gpu_data  # ✅ Fetch latest GPU data from thread
    return f"CPU: {cpu_usage}% | RAM: {ram_usage}% | {gpu_usage}"

# Function to monitor GPU usage (runs in background)
gpu_data = "GPU: N/A"

def update_gpu_usage():
    """Fetch GPU usage using powermetrics."""
    global gpu_data
    while True:
        try:
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n1"],
                capture_output=True, text=True
            )
            gpu_usage = "GPU: N/A"

            for line in result.stdout.split("\n"):
                if "GPU Power" in line:
                    gpu_usage = f"GPU: {line.split(':')[1].strip()} W"

            gpu_data = gpu_usage  # ✅ Only GPU tracking
        except:
            gpu_data = "GPU: Error"

        time.sleep(5)  # ✅ Update every 5 seconds

# Start background thread for GPU monitoring
threading.Thread(target=update_gpu_usage, daemon=True, name="GPU_Usage_Monitor").start()

# Function to interact with the LLM
def chat_with_model(user_input, chat_history):
    selected_model = select_model(user_input)
    chat_messages = chat_history + [{"role": "user", "content": user_input}]

    # ✅ Immediately update UI with user input
    yield chat_messages, f"Model Used: {selected_model}", "", get_system_resources(), None  

    # ✅ Ensure `generated_image` is always initialized
    generated_image = None  

    # ✅ FIX: Use Stability AI API for image generation
    if selected_model == "stability-api":  # ✅ Check for Stability AI API
        generated_image = generate_image(user_input)

        if generated_image and not generated_image.startswith("Error"):  # ✅ Ensure valid image path
            yield chat_messages, f"Model Used: Stability AI", "", get_system_resources(), generated_image  
        else:
            yield chat_messages, f"Model Used: Stability AI", generated_image, get_system_resources(), None

        return  # ✅ Exit since image requests don’t need an LLM response

    # ✅ FIX: Use `ollama.generate()` correctly for text models
    try:
        response = ollama.generate(model=selected_model, prompt=user_input)

        # ✅ Handle different response structures
        if isinstance(response, dict):
            answer = response.get("message", {}).get("content", "").strip() or response.get("response", "").strip()
        elif isinstance(response, str):
            answer = response.strip()
        elif hasattr(response, "response"):
            answer = getattr(response, "response", "").strip()
        else:
            answer = "⚠️ No response generated."

        # ✅ Remove <think>...</think> from DeepSeek-R1 responses
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        # ✅ Force DeepSeek-R1 to provide a structured answer
        if selected_model == "deepseek-r1:latest" and not answer:
            user_input = f"Please provide a direct response: {user_input}"
            response = ollama.generate(model=selected_model, prompt=user_input)
            answer = response.get("message", {}).get("content", "").strip() or response.get("response", "").strip()

        # ✅ Handle empty response
        if not answer:
            answer = "⚠️ No response generated."

        chat_messages.append({"role": "assistant", "content": answer})

        # ✅ Save response to chat history
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chats (user_prompt, model_response, model_used) VALUES (?, ?, ?)",
                    (user_input, answer, selected_model))
        conn.commit()
        conn.close()

        yield chat_messages, f"Model Used: {selected_model}", "", get_system_resources(), None  

    except Exception as e:
        yield chat_messages, f"⚠️ Error: {str(e)}", "", get_system_resources(), None  

# Function to handle file uploads
def handle_file_upload(file):
    if file is None:
        return "No file uploaded."

    # Ensure the upload directory exists
    upload_dir = "./uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)  # ✅ Create directory if it doesn't exist

    # ✅ If `file` is already a string (path), use it directly
    if isinstance(file, str):
        saved_path = os.path.join(upload_dir, os.path.basename(file))
        os.rename(file, saved_path)  # ✅ Move the file instead of reading/writing it

    else:
        file_path = file.name
        saved_path = os.path.join(upload_dir, os.path.basename(file_path))

        # ✅ Read file content properly if it's a file object
        with open(saved_path, "wb") as f:
            f.write(file.read())  # ✅ Ensure `file.read()` is only used when `file` is a file object

    file_size = os.path.getsize(saved_path)  # Get file size after saving
    return f"Uploaded File: {os.path.basename(saved_path)} | Size: {round(file_size / 1024, 2)} KB"

# Function to load a selected previous chat
def load_selected_chat(selected_chat):
    if not selected_chat or selected_chat == "No previous chats":
        return [{"role": "assistant", "content": "No messages available in this chat."}]

    chat_id = int(selected_chat.split(":")[0].replace("Chat ", ""))
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_prompt, model_response FROM chats WHERE id = ?", (chat_id,))
    chats = cursor.fetchall()
    conn.close()

    # ✅ Ensure the output is properly formatted as a list of dictionaries
    formatted_chat = []
    for user_msg, assistant_msg in chats:
        formatted_chat.append({"role": "user", "content": user_msg})
        formatted_chat.append({"role": "assistant", "content": assistant_msg if assistant_msg.strip() else "No response available."})

    if not formatted_chat:
        formatted_chat = [{"role": "assistant", "content": "No messages found for this chat."}]

    print("✅ Loaded selected chat messages:", formatted_chat)  # Debugging output
    return formatted_chat

# Function to delete a selected chat from history
def delete_selected_chat(selected_chat):
    if not selected_chat or selected_chat == "No previous chats":
        return [{"role": "assistant", "content": "Chat deleted. No messages available."}], "", gr.update(choices=load_chat_history(), value="No previous chats")

    chat_id = int(selected_chat.split(":")[0].replace("Chat ", ""))

    # Delete the selected chat from the database
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()

    # Load updated chat history
    updated_chat_list = load_chat_history()
    new_selected_chat = updated_chat_list[0] if updated_chat_list else "No previous chats"

    # Load the newly selected chat (if any)
    loaded_chat = load_selected_chat(new_selected_chat)

    # Ensure chatbot messages are properly formatted
    chatbot_messages = []
    for user_msg, assistant_msg in loaded_chat:
        chatbot_messages.append({"role": "user", "content": user_msg})
        chatbot_messages.append({"role": "assistant", "content": assistant_msg if assistant_msg.strip() else "No response available."})

    # If no messages remain, provide a default assistant response
    if not chatbot_messages:
        chatbot_messages = [{"role": "assistant", "content": "Chat deleted. Starting a new conversation."}]

    print("✅ Chat deleted. Returning properly formatted chatbot messages:", chatbot_messages)  # Debugging log

    return chatbot_messages, "", gr.update(choices=updated_chat_list, value=new_selected_chat)

# Function to clear chat history and start a new chat
def new_chat():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (user_prompt, model_response, model_used) VALUES (?, ?, ?)",
                   ("New chat started", "", "None"))
    conn.commit()
    conn.close()
    updated_chat_list = load_chat_history()
    return [], "", gr.update(choices=updated_chat_list, value=updated_chat_list[0])

# Function to stop ongoing processing
def stop_response():
    print("🛑 Stopping response processing...")  # ✅ Debugging output
    return [], "", "", gr.update(), None  # ✅ Ensure exactly 5 outputs

# Function to download the generated image
def download_image():
    file_path = "generated_image.png"
    if pathlib.Path(file_path).exists():
        return gr.File(file_path)  # ✅ Ensure it's handled as a file
    else:
        return "Error: No image generated yet."

# UI Design
with gr.Blocks() as ui:
    gr.Markdown("# Hi Farzad :) This is your Local AI Chat (Auto Model Selection) 🤖")

    # Row for chat history selection and management
    with gr.Row():
        chat_list = gr.Dropdown(choices=load_chat_history(), label="Previous Chats", interactive=True, allow_custom_value=True, scale=2)
        delete_chat_btn = gr.Button("🗑️ Delete", scale=1)
        new_chat_btn = gr.Button("➕ New", scale=1)
        attach_file_btn = gr.Button("📎 Attach File", scale=1)  # ✅ Styled as a button
        file_info = gr.Textbox(label="File Info", interactive=False, scale=2, show_label=False)  # ✅ Display file name

    # Hidden File Upload Component
    hidden_file_upload = gr.File(type="filepath", interactive=True, visible=False)  # ✅ Correctly defined

    # Chatbot UI
    with gr.Row():
        chatbot = gr.Chatbot(label="Chatbot", scale=3, type="messages")  # ✅ Ensuring OpenAI-style messages

    # Input Box
    with gr.Row():
        textbox = gr.Textbox(label="Your Message", placeholder="Type your question here...", scale=2)

    # Model Used & System Resources
    with gr.Row():
        model_used = gr.Textbox(label="Model Used", interactive=False, scale=1)
        system_resources = gr.Textbox(label="System Resources", interactive=False, scale=1)

    # Image Generation & Download
    with gr.Row():
        image_output = gr.Image(label="Generated Image", type="pil", interactive=False)

    # Buttons at the bottom
    with gr.Row():
        send_button = gr.Button("Send", scale=1)
        cancel_button = gr.Button("Cancel", scale=1)

    # ✅ Event bindings
    chat_list.change(load_selected_chat, inputs=[chat_list], outputs=[chatbot])
    send_button.click(chat_with_model, inputs=[textbox, chatbot],outputs=[chatbot, model_used, textbox, system_resources, image_output])
    textbox.submit(chat_with_model, inputs=[textbox, chatbot], outputs=[chatbot, model_used, textbox, system_resources, image_output])
    cancel_button.click(stop_response, outputs=[chatbot, model_used, textbox, system_resources, image_output])
    new_chat_btn.click(new_chat, outputs=[chatbot, model_used, chat_list])
    delete_chat_btn.click(delete_selected_chat, inputs=[chat_list], outputs=[chatbot, model_used, chat_list])
    attach_file_btn.click(lambda: gr.update(visible=True), outputs=[hidden_file_upload])  # ✅ Opens file picker
    hidden_file_upload.change(handle_file_upload, inputs=[hidden_file_upload], outputs=[file_info])  # ✅ Processes file

ui.launch()
