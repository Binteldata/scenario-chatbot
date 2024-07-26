import warnings
import customtkinter as ctk
from openai import OpenAI
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import threading
import torch
import json
import os
import speech_recognition as sr


# Suppress specific FutureWarning from torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class TextToSpeechApp:
    def __init__(self, api_key):
        self.api_key = api_key
        self.setup_openai()
        self.setup_tts()
        self.load_scenarios()
        self.setup_ui()
        self.message_queue = []  # Queue for UI updates (optional)

    def setup_openai(self):
        # Initialize OpenAI API with the provided key
        self.client = OpenAI(api_key=self.api_key)

    def setup_tts(self):
        # Initialize the Text-to-Speech model
        model_name = "tts_models/en/vctk/vits"
        self.tts = TTS(model_name=model_name, progress_bar=False, gpu=torch.cuda.is_available())

    def load_scenarios(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        roles_file = os.path.join(script_path, 'scenarios.json')
        with open(roles_file, 'r') as file:
            self.scenarios_data = json.load(file)

    def setup_ui(self):
        # Set up the user interface using customtkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create the main window
        self.root = ctk.CTk()
        self.root.title("Scenario Chat Bot")
        self.root.geometry("800x600")

        # Create the main frame
        self.frame = ctk.CTkFrame(master=self.root)
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Create the chat log display area
        self.chat_log = ctk.CTkTextbox(master=self.frame, width=700, height=400, font=("Roboto", 14))
        self.chat_log.pack(pady=10, padx=10)
        self.chat_log.configure(state="disabled")

        # Create the user input field
        self.user_input = ctk.CTkEntry(master=self.frame, width=550, height=40, font=("Roboto", 14))
        self.user_input.pack(pady=10, padx=10)

        # Create a frame for buttons
        self.button_frame = ctk.CTkFrame(master=self.frame)
        self.button_frame.pack(pady=10, padx=10)

        # Create the send button
        self.send_button = ctk.CTkButton(master=self.button_frame, text="Send", command=self.send_message, width=100, height=40)
        self.send_button.pack(side="left", padx=(0, 10))

        # Create the voice input button
        self.voice_input_button = ctk.CTkButton(master=self.button_frame, text="Voice Input", command=self.handle_voice_input, width=90, height=40)
        self.voice_input_button.pack(side="left", padx=10)

        # Create the voice selection dropdown
        self.voice_var = ctk.StringVar(value="Voice Changer")
        self.voice_menu = ctk.CTkOptionMenu(master=self.button_frame, values=self.tts.speakers, variable=self.voice_var, width=100, height=40)
        self.voice_menu.pack(side="left")

        # Create the scenario selection dropdown
        self.scenario_var = ctk.StringVar(value="Select Scenario")
        self.scenario_menu = ctk.CTkOptionMenu(master=self.button_frame, values=list(self.scenarios_data.keys()), variable=self.scenario_var, width=150, height=40)
        self.scenario_menu.pack(side="left")

    def send_message(self):
        # Get the user's message and clear the input field
        user_message = self.user_input.get()
        if user_message:
            self.update_chat_log("You: " + user_message)
            self.user_input.delete(0, "end")
            # Start a new thread to get the AI response
            threading.Thread(target=self.get_ai_response, args=(user_message,)).start()

    def get_ai_response(self, user_message):
        # Get the selected scenario
        scenario_name = self.scenario_var.get()
        scenario = self.scenarios_data.get(scenario_name, "")
        if not scenario:
            # Handle invalid scenario case
            self.update_chat_log("Invalid scenario selected. Please choose a valid scenario.")
            return

        messages = [{"role": "system", "content": scenario}, {"role": "user", "content": user_message}]

        # Get a response from the OpenAI API
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            ai_message = response.choices[0].message.content
        except Exception as e:
            self.update_chat_log(f"Error: {str(e)}")
            return

        # Update the chat log potentially using a queue
        self.update_chat_log("AI: " + ai_message)
        self.speak(ai_message)

    def update_chat_log(self, message):
        # Add a new message to the chat log
        self.message_queue.append(message)
        self.process_message_queue()

    def process_message_queue(self):
        while self.message_queue:
            message = self.message_queue.pop(0)
            self.chat_log.configure(state="normal")
            self.chat_log.insert("end", message + "\n\n")
            self.chat_log.see("end")
            self.chat_log.configure(state="disabled")

    def speak(self, text):
        # Convert text to speech and play it
        speaker = self.voice_var.get()
        wav = self.tts.tts(text=text, speaker=speaker)
        sd.play(np.array(wav), samplerate=22050)
        #sd.wait()

    def listen_for_speech(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
        
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def handle_voice_input(self):
        user_message = self.listen_for_speech()
        if user_message:
            self.user_input.delete(0, "end")
            self.user_input.insert(0, user_message)
            self.send_message()

    def run(self):
        # Start the main event loop
        self.root.mainloop()

if __name__ == "__main__":
    # Example usage
    api_key = "ADD API KEY"
    app = TextToSpeechApp(api_key)
    app.run()