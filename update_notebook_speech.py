#!/usr/bin/env python3
"""
Update the notebook with improved speech recognition
"""

import json

# Read the notebook
with open('sales_voice_agent.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the listen method in the VoiceAgent class
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'def listen(self):' in str(cell['source']):
        # Update the listen method
        source_lines = cell['source']
        new_source = []
        
        for line in source_lines:
            if 'def listen(self):' in line:
                new_source.append(line)
                new_source.append('        with sr.Microphone() as source:\n')
                new_source.append('            # Adjust microphone for ambient noise\n')
                new_source.append('            self.listener.adjust_for_ambient_noise(source, duration=0.5)\n')
                new_source.append('            print("🎤 Listening... (speak naturally, I\'ll wait for you to finish)")\n')
                new_source.append('            \n')
                new_source.append('            # Increase timeout and phrase_time_limit for more patient listening\n')
                new_source.append('            audio = self.listener.listen(\n')
                new_source.append('                source, \n')
                new_source.append('                timeout=10,  # Wait up to 10 seconds for speech to start\n')
                new_source.append('                phrase_time_limit=15,  # Allow up to 15 seconds for a complete phrase\n')
                new_source.append('                snowboy_configuration=None\n')
                new_source.append('            )\n')
                new_source.append('        try:\n')
                new_source.append('            text = self.listener.recognize_google(audio)\n')
                new_source.append('            print(f"👂 You said: {text}")\n')
                new_source.append('            return text\n')
                new_source.append('        except sr.WaitTimeoutError:\n')
                new_source.append('            print("⏰ No speech detected within timeout period")\n')
                new_source.append('            return ""\n')
                new_source.append('        except sr.UnknownValueError:\n')
                new_source.append('            print("❌ Could not understand the audio")\n')
                new_source.append('            return ""\n')
                new_source.append('        except Exception as e:\n')
                new_source.append('            print(f"❌ Error: {e}")\n')
                new_source.append('            return ""\n')
                break
            elif 'with sr.Microphone() as source:' in line:
                # Skip the old implementation
                continue
            elif 'print("🎤 Listening...")' in line:
                continue
            elif 'audio = self.listener.listen(source)' in line:
                continue
            elif 'try:' in line and 'text = self.listener.recognize_google(audio)' in str(source_lines[source_lines.index(line)+1]):
                continue
            elif 'text = self.listener.recognize_google(audio)' in line:
                continue
            elif 'print(f"👂 You said: {text}")' in line:
                continue
            elif 'return text' in line and 'except Exception as e:' in str(source_lines[source_lines.index(line)+1]):
                continue
            elif 'except Exception as e:' in line:
                continue
            elif 'print(f"❌ Could not understand: {e}")' in line:
                continue
            elif 'return ""' in line and source_lines.index(line) < len(source_lines) - 1 and 'def speak(self, text):' in str(source_lines[source_lines.index(line)+1]):
                continue
            else:
                new_source.append(line)
        
        cell['source'] = new_source
        break

# Write the updated notebook
with open('sales_voice_agent.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Updated notebook with improved speech recognition!") 