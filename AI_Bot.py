#used google colab
#import statements
!pip install transformers ipywidgets
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
from ipywidgets import widgets, VBox, HBox
from IPython.display import display, clear_output
#The set_seed() function in the Hugging Face transformers library sets the random seed for various random number generators used within the framework. This includes randomness in model weight initialization, data shuffling, and sampling during text generation.
set_seed(42)
#A text generation pipeline initialized with distilgpt2, a smaller and faster version of GPT-2.
#pad_token_id=50256 ensures padding is handled correctly for the model.
generator = pipeline("text-generation", model="distilgpt2", pad_token_id=50256) #is a variant of gpt2
#GPT-2 does not give the best response lol
user_inputs = []
bot_responses = []
#response function
def generate_response(prompt, max_length=100):
    """Generate a response using distilGPT-2 with improved parameters."""
    response = generator(
        prompt,
        max_length=max_length,
        min_length=20,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        truncation=True
    )[0]["generated_text"]

    bot_reply = response[len(prompt):].strip()
    if len(bot_reply) < 5:
        return "I couldn't think of anything interesting to say! Let's try again."
    return bot_reply
#chat_handler code
def chat_handler(change):
    clear_output(wait=True)
    user_message = user_input.value.strip()
    if user_message:
        user_inputs.append(user_message)
        bot_reply = generate_response(user_message)
        bot_responses.append(bot_reply)

        print("💬 **Your Message:**", user_message)
        print("🤖 **Chatbot's Reply:**", bot_reply)
#display
user_input = widgets.Text(placeholder="Type your message here...", description="You:")
send_button = widgets.Button(description="Send", button_style="success")
send_button.on_click(chat_handler)

display(widgets.HTML("🤖 Small Language Model Chatbot"))
display(HBox([user_input, send_button]))





