import os
import json
import pyttsx3
import speech_recognition as sr
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import datetime
from calendar_helper import create_event

DOC_PATH = "documents.json"
INDEX_PATH = "vector.index"
EMBEDDINGS_PATH = "embeddings.pkl"
CALENDAR_PATH = "calendar.txt"


from calendar_helper import create_event
import datetime

# Example usage:
now = datetime.datetime.now() + datetime.timedelta(hours=1)
create_event("Call with Customer", now, duration_minutes=15)

class RAG:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def build_index(self, docs):
        self.documents = docs
        embeddings = self.model.encode(docs)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
        with open(EMBEDDINGS_PATH, "wb") as f:
            import pickle
            pickle.dump(docs, f)

    def load_index(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(EMBEDDINGS_PATH, "rb") as f:
            import pickle
            self.documents = pickle.load(f)

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in I[0]]

class VoiceAgent:
    def __init__(self):
        self.rag = RAG()
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float32)
        self.llm.eval()
        self.tts = pyttsx3.init()
        self.listener = sr.Recognizer()

    def listen(self):
        with sr.Microphone() as source:
            print("🎤 Listening...")
            audio = self.listener.listen(source)
        try:
            text = self.listener.recognize_google(audio)
            print(f"👂 You said: {text}")
            return text
        except Exception as e:
            print(f"❌ Could not understand: {e}")
            return ""

    def speak(self, text):
        print(f"🤖 Assistant: {text}")
        self.tts.say(text)
        self.tts.runAndWait()

    def answer(self, query):
        context = "\n".join(self.rag.retrieve(query))
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def parse_schedule_request(self, user_input):
        """Parse user input to extract scheduling information"""
        user_input_lower = user_input.lower()
        
        # Default values
        event_title = "Sales Call"
        duration = 30
        start_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        
        # Extract event title
        if "call" in user_input_lower:
            if "demo" in user_input_lower:
                event_title = "Product Demo Call"
            elif "sales" in user_input_lower:
                event_title = "Sales Call"
            elif "meeting" in user_input_lower:
                event_title = "Sales Meeting"
            else:
                event_title = "Call"
        
        # Extract duration
        duration_match = re.search(r'(\d+)\s*(min|minute|minutes|hour|hours)', user_input_lower)
        if duration_match:
            value = int(duration_match.group(1))
            unit = duration_match.group(2)
            if unit in ['hour', 'hours']:
                duration = value * 60
            else:
                duration = value
        
        # Extract time information
        time_patterns = [
            r'tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
            r'today\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
            r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
            r'in\s+(\d+)\s*(hour|hours|minute|minutes)',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                if 'tomorrow' in pattern:
                    start_time = datetime.datetime.now() + datetime.timedelta(days=1)
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if match.group(2) else 0
                    ampm = match.group(3)
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    elif ampm == 'am' and hour == 12:
                        hour = 0
                    start_time = start_time.replace(hour=hour, minute=minute)
                    break
                elif 'today' in pattern:
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if match.group(2) else 0
                    ampm = match.group(3)
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    elif ampm == 'am' and hour == 12:
                        hour = 0
                    start_time = datetime.datetime.now().replace(hour=hour, minute=minute)
                    if start_time < datetime.datetime.now():
                        start_time += datetime.timedelta(days=1)
                    break
                elif 'in' in pattern:
                    value = int(match.group(1))
                    unit = match.group(2)
                    if unit in ['hour', 'hours']:
                        start_time = datetime.datetime.now() + datetime.timedelta(hours=value)
                    else:
                        start_time = datetime.datetime.now() + datetime.timedelta(minutes=value)
                    break
                else:
                    # Just time specified, assume today
                    hour = int(match.group(1))
                    minute = int(match.group(2)) if match.group(2) else 0
                    ampm = match.group(3)
                    if ampm == 'pm' and hour != 12:
                        hour += 12
                    elif ampm == 'am' and hour == 12:
                        hour = 0
                    start_time = datetime.datetime.now().replace(hour=hour, minute=minute)
                    if start_time < datetime.datetime.now():
                        start_time += datetime.timedelta(days=1)
                    break
        
        return event_title, start_time, duration

    def schedule_event(self, user_input):
        """Create a calendar event based on user input"""
        try:
            event_title, start_time, duration = self.parse_schedule_request(user_input)
            
            # Create the calendar event
            event_link = create_event(
                summary=event_title,
                start_time=start_time,
                duration_minutes=duration,
                description=f"Automatically scheduled based on: {user_input}"
            )
            
            # Format time for speech
            time_str = start_time.strftime("%I:%M %p on %B %d")
            
            return f"I've scheduled a {duration}-minute {event_title} for {time_str}. The event has been added to your calendar."
            
        except Exception as e:
            print(f"Error creating calendar event: {e}")
            return "I'm sorry, I couldn't create the calendar event. Please try again."

    def add_to_calendar(self, info):
        with open(CALENDAR_PATH, "a") as f:
            f.write(info + "\n")

    def run(self):
        if not os.path.exists(INDEX_PATH):
            print("⚙️ Building RAG index...")
            with open(DOC_PATH) as f:
                docs = json.load(f)
            self.rag.build_index(docs)
        else:
            self.rag.load_index()

        self.speak("Hello! I am the Sales Executive of Hyprbots. Im here to explain about our products and services.")
        while True:
            user_input = self.listen()
            if not user_input:
                continue
            if "exit" in user_input.lower() or "bye" in user_input.lower():
                self.speak(" Thank you for your patience! Have a Good Day!")
                break
            
            # Check for scheduling requests first
            if "schedule" in user_input.lower() or "book" in user_input.lower() or "appointment" in user_input.lower():
                response = self.schedule_event(user_input)
                self.speak(response)
                # Also log to text file as backup
                self.add_to_calendar(f"Calendar event created: {user_input}")
            else:
                answer = self.answer(user_input)
                self.speak(answer)

if __name__ == "__main__":
    agent = VoiceAgent()
    agent.run()
