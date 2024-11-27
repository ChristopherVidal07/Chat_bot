from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests

# Flask app
app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.config.pad_token_id = tokenizer.eos_token_id

# API keys (replace with your own keys)
WEATHER_API_KEY = "0bb7192e77027ed29dc047322617e01c"
NEWS_API_KEY = "dee4eed1e82d417ca60256fd3bbdfeff"

# Joke API function
def get_joke():
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke")
        if response.status_code == 200:
            joke = response.json()
            return f"{joke['setup']} - {joke['punchline']}"
        else:
            return "I couldn't fetch a joke at the moment. Try again later!"
    except Exception as e:
        return f"Oops! Something went wrong while fetching the joke: {str(e)}"

# Weather API function
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={WEATHER_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            return f"The weather in {city} is currently {description} with a temperature of {temperature}°C."
        else:
            return "I couldn't fetch the weather for that location. Please check the city name."
    except Exception as e:
        return f"Oops! Something went wrong while fetching the weather: {str(e)}"

# News API function
def get_news():
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()["articles"][:5]  # Fetch top 5 articles
            return "\n".join([f"- {article['title']}" for article in articles])
        else:
            return "I couldn't fetch the news at the moment. Please try again later."
    except Exception as e:
        return f"Oops! Something went wrong while fetching the news: {str(e)}"

# Chat function using DialoGPT
def chat_with_pipeline(user_input, chat_history_ids=None):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = (
        torch.cat([chat_history_ids, inputs], dim=-1) if chat_history_ids is not None else inputs
    )
    response_ids = model.generate(
        chat_history_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(chat_history_ids),
    )
    response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Route to serve the homepage
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint for chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("user_input", "").lower()
    if not user_input:
        return jsonify({"response": "Please provide input text."}), 400

    # Handle specific commands
    if "joke" in user_input:
        return jsonify({"response": get_joke()})
    elif "weather" in user_input:
        # Extract city name from user input
        city = user_input.split("in")[-1].strip()
        return jsonify({"response": get_weather(city)})
    elif "news" in user_input:
        return jsonify({"response": get_news()})

    # Fallback to chatbot response
    response = chat_with_pipeline(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
