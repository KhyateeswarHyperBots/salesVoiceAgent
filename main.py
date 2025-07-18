import os
import json
import pyttsx3
import speech_recognition as sr
import faiss
import ollama
from sentence_transformers import SentenceTransformer
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
# create_event("Call with Customer", now, duration_minutes=15)

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
        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(EMBEDDINGS_PATH, "rb") as f:
                import pickle
                self.documents = pickle.load(f)
            print(f"✅ Loaded RAG index with {len(self.documents)} documents")
        except Exception as e:
            print(f"❌ Error loading RAG index: {e}")
            print("🔄 Rebuilding index...")
            # If loading fails, rebuild the index
            with open(DOC_PATH) as f:
                docs = json.load(f)
            self.build_index(docs)

    def retrieve(self, query, top_k=3):
        if self.index is None:
            print("⚠️ RAG index not loaded, attempting to load...")
            self.load_index()
            if self.index is None:
                return ["No documents available for retrieval."]
        
        query_vec = self.model.encode([query])
        D, I = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in I[0]]

class VoiceAgent:
    def __init__(self):
        self.rag = RAG()
        self.model_name = "llama3.2:3b"
        self.tts = pyttsx3.init()
        self.listener = sr.Recognizer()

    def listen(self):
        with sr.Microphone() as source:
            # Adjust microphone for ambient noise
            self.listener.adjust_for_ambient_noise(source, duration=0.5)
            print("🎤 Listening... (speak naturally, I'll wait for you to finish)")
            
            # Increase timeout and phrase_time_limit for more patient listening
            audio = self.listener.listen(
                source, 
                timeout=10,  # Wait up to 10 seconds for speech to start
                phrase_time_limit=15,  # Allow up to 15 seconds for a complete phrase
                snowboy_configuration=None
            )
        try:
            text = self.listener.recognize_google(audio)
            print(f"👂 You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("⏰ No speech detected within timeout period")
            return ""
        except sr.UnknownValueError:
            print("❌ Could not understand the audio")
            return ""
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""

    def speak(self, text):
        print(f"🤖 Assistant: {text}")
        self.tts.say(text)
        self.tts.runAndWait()

    def answer(self, query):
        retrieved_docs = self.rag.retrieve(query)
        # Convert documents to strings if they're dictionaries
        context_parts = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                content = doc.get('content', str(doc))
            else:
                content = str(doc)
            context_parts.append(content)
        
        context = "\n".join(context_parts)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return "I'm sorry, I couldn't process your request right now."

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
