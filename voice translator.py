import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from googletrans import Translator
from datetime import datetime

recognizer = sr.Recognizer()
translator = Translator()

START_TIME = "14:30"
END_TIME = "22:00"

def is_time_allowed():
    current_time = datetime.now().strftime("%H:%M")
    return START_TIME <= current_time <= END_TIME

def translate_audio():
    if not is_time_allowed():
        output_text.set("🌙 Taking rest, see you tomorrow!")
        return

    try:
        with sr.Microphone() as source:
            output_text.set("🎧 Listening... Speak now!")
            root.update()
            audio_data = recognizer.listen(source, timeout=5)
            output_text.set("🛠 Processing audio...")
            root.update()

            text = recognizer.recognize_google(audio_data, language="en-IN")
            print("Recognized English:", text)

            translated = translator.translate(text, src='en', dest='hi')
            output_text.set(f"🗣 Hindi: {translated.text}")

    except sr.UnknownValueError:
        output_text.set("⚠️ Didn't catch that. Please repeat clearly.")
    except sr.RequestError:
        output_text.set("🚫 Error connecting to speech service.")
    except Exception as e:
        output_text.set(f"❌ Error: {str(e)}")

root = tk.Tk()
root.title("🎤 Voice Translator (English ➝ Hindi)")
root.geometry("500x300")
root.resizable(False, False)

canvas = tk.Canvas(root, width=500, height=300)
canvas.pack(fill="both", expand=True)

for i in range(300):
    r = int(240 - (i / 300) * 40)
    g = int(240 - (i / 300) * 80)
    b = int(255 - (i / 300) * 60)
    color = f'#{r:02x}{g:02x}{b:02x}'
    canvas.create_line(0, i, 500, i, fill=color)


frame = tk.Frame(root, bg="#ffffff", bd=2, relief="flat")
frame.place(relx=0.5, rely=0.5, anchor="center", width=400, height=220)

title_label = tk.Label(frame, text="🎤 English ➝ Hindi Translator", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#333333")
title_label.pack(pady=15)

def on_enter(e):
    translate_button.config(bg="#388E3C")

def on_leave(e):
    translate_button.config(bg="#4CAF50")

translate_button = tk.Button(frame, text="Start Listening", font=("Helvetica", 14, "bold"),
                             bg="#4CAF50", fg="white", activebackground="#45a049",
                             relief="flat", padx=10, pady=5, command=translate_audio)
translate_button.pack(pady=10)
translate_button.bind("<Enter>", on_enter)
translate_button.bind("<Leave>", on_leave)

output_text = tk.StringVar()
output_label = tk.Label(frame, textvariable=output_text, font=("Helvetica", 13),
                        wraplength=350, bg="#f5f5f5", fg="#333333",
                        padx=10, pady=10, relief="solid", bd=1)
output_label.pack(pady=15, fill="x", padx=10)

root.mainloop()
