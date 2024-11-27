from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import random
import requests

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.config.pad_token_id = tokenizer.eos_token_id  # Explicitly set pad_token_id

# Initialize the pipeline
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Memory to store user information
memory = {}

# Update memory
def update_memory(key, value):
    memory[key] = value

# Get memory
def get_memory(key):
    return memory.get(key, None)

# Chat function using the pipeline
def chat_with_pipeline(user_input, chat_history_ids=None):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = (
        torch.cat([chat_history_ids, inputs], dim=-1) if chat_history_ids is not None else inputs
    )
    
    # Generate a response
    response_ids = model.generate(
        chat_history_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(chat_history_ids),
    )
    response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Fetch trivia questions from OpenTDB
def fetch_trivia_questions(amount=5, category=None, difficulty=None):
    base_url = "https://opentdb.com/api.php"
    params = {"amount": amount}
    if category:
        params["category"] = category
    if difficulty:
        params["difficulty"] = difficulty

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["response_code"] == 0:  # Successful response
                return data["results"]
            else:
                print("Chatbot: No trivia questions found. Try again!")
                return None
        else:
            print("Chatbot: Failed to fetch trivia questions.")
            return None
    except Exception as e:
        print(f"Chatbot: Error fetching trivia: {e}")
        return None

# Trivia game logic
def trivia_game():
    print("Chatbot: Choose a difficulty level: easy, medium, or hard.")
    difficulty = input("You: ").strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        print("Chatbot: Invalid choice, defaulting to 'medium'.")
        difficulty = "medium"

    print("Chatbot: Choose a category (type a number):")
    categories = {
        9: "General Knowledge",
        18: "Science: Computers",
        23: "History",
        22: "Geography",
        12: "Music",
    }
    for k, v in categories.items():
        print(f"{k}: {v}")
    try:
        category = int(input("You: ").strip())
        if category not in categories:
            print("Chatbot: Invalid choice, defaulting to General Knowledge.")
            category = 9
    except ValueError:
        print("Chatbot: Invalid input, defaulting to General Knowledge.")
        category = 9

    questions = fetch_trivia_questions(amount=5, category=category, difficulty=difficulty)
    if not questions:
        return "I couldn't fetch trivia questions. Please try again later!"

    score = 0
    for idx, q in enumerate(questions, 1):
        print(f"\nQuestion {idx}: {q['question']}")
        options = q["incorrect_answers"] + [q["correct_answer"]]
        random.shuffle(options)

        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            user_answer = int(input("Your answer (1/2/3/4): ").strip())
            if options[user_answer - 1] == q["correct_answer"]:
                print("Chatbot: Correct! 🎉")
                score += 1
            else:
                print(f"Chatbot: Incorrect. The correct answer was: {q['correct_answer']}")
        except (ValueError, IndexError):
            print("Chatbot: Invalid input. Moving to the next question.")

    return f"\nYour final score is {score}/{len(questions)}! Thanks for playing."

# Fetch a joke from Joke API
def get_joke():
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke")
        if response.status_code == 200:
            joke = response.json()
            return f"{joke['setup']} - {joke['punchline']}"
        else:
            return "Chatbot: I couldn't fetch a joke at the moment. Try again later!"
    except Exception as e:
        return f"Chatbot: Oops! Something went wrong while fetching the joke: {str(e)}"

# Main chatbot loop
print("Chatbot: Hi! Type 'bye' to exit.")
chat_history_ids = None
while True:
    user_input = input("You: ").strip()

    # Exit
    if user_input.lower() == "bye":
        print("Chatbot: Goodbye!")
        break

    # Handle user introduction
    elif user_input.lower().startswith("my name is"):
        name = user_input[11:].strip()
        if name:
            update_memory("name", name)
            print(f"Chatbot: Nice to meet you, {name}!")
        else:
            print("Chatbot: I didn't catch your name. Please repeat it.")

    # Personalized greetings if name is remembered
    elif user_input.lower() in ["hello", "hi"] and get_memory("name"):
        print(f"Chatbot: Hello, {get_memory('name')}! How can I assist you today?")

    # Handle trivia game
    elif user_input.lower() == "let's play trivia":
        print(f"Chatbot: {trivia_game()}")

    # Fetch and tell a joke
    elif user_input.lower() in ["tell me a joke", "joke"]:
        print(f"Chatbot: {get_joke()}")

    # Fallback to Hugging Face pipeline for responses
    else:
        response, chat_history_ids = chat_with_pipeline(user_input, chat_history_ids)
        print(f"Chatbot: {response}")
