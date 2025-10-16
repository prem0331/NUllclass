
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from transformers import MarianMTModel, MarianTokenizer
import torch
import re

class DualLanguageTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Language Translator - English to French & Hindi")
        self.root.geometry("800x600")
        
        self.supported_langs = {
            'French': 'Helsinki-NLP/opus-mt-en-fr',
            'Hindi': 'Helsinki-NLP/opus-mt-en-hi'
        }
        
        self.models = {}
        self.tokenizers = {}
        self.load_models()
        
        self.setup_gui()
        
        self.translate_results = {'French': '', 'Hindi': '', 'Summary': ''}

    def load_models(self):
        """Load models in a background thread to avoid freezing GUI."""
        def load():
            for lang, model_name in self.supported_langs.items():
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    self.tokenizers[lang] = tokenizer
                    self.models[lang] = model
                    print(f"Loaded {lang} model successfully.")
                except Exception as e:
                    messagebox.showerror("Model Load Error", f"Failed to load {lang}: {str(e)}")
                    print(f"Error loading {lang}: {e}")
        
        threading.Thread(target=load, daemon=True).start()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="English Input (10+ characters per word):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_text = scrolledtext.ScrolledText(main_frame, width=60, height=6)
        self.input_text.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(main_frame, text="Clear Input", command=self.clear_input).grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Button(main_frame, text="Translate", command=self.start_translation).grid(row=2, column=1, sticky=tk.E, padx=5)

        ttk.Label(main_frame, text="Translation Results").grid(row=3, column=0, sticky=tk.W, pady=(20, 5))
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        
        french_frame = ttk.Frame(self.notebook)
        self.notebook.add(french_frame, text="French")
        self.french_text = scrolledtext.ScrolledText(french_frame, width=60, height=10)
        self.french_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        hindi_frame = ttk.Frame(self.notebook)
        self.notebook.add(hindi_frame, text="Hindi")
        self.hindi_text = scrolledtext.ScrolledText(hindi_frame, width=60, height=10)
        self.hindi_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        self.summary_text = scrolledtext.ScrolledText(summary_frame, width=60, height=10)
        self.summary_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Copy All", command=self.copy_all).pack(side=tk.LEFT, padx=5)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

    def clear_input(self):
        self.input_text.delete(1.0, tk.END)

    def clear_all(self):
        self.input_text.delete(1.0, tk.END)
        self.french_text.delete(1.0, tk.END)
        self.hindi_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        self.translate_results = {'French': '', 'Hindi': '', 'Summary': ''}

    def filter_words_by_length(self, text):
        """Filter words that have 10 or more characters"""
        words = re.findall(r'\b\w+\b', text)  # Extract all words
        long_words = [word for word in words if len(word) >= 10]
        short_words = [word for word in words if len(word) < 10]
        
        return long_words, short_words

    def start_translation(self):
        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("No Input", "Please enter English text to translate.")
            return
        
        long_words, short_words = self.filter_words_by_length(text)
        
        if short_words:
            short_word_msg = " | ".join(short_words[:5]) 
            if len(short_words) > 5:
                short_word_msg += f" and {len(short_words) - 5} more"
            
            messagebox.showwarning("Short Words Detected", 
                                 f"Words with <10 characters found: {short_word_msg}\n\n"
                                 f"These words need to be 'uploaded again'.\n"
                                 f"Only words with 10+ characters will be translated:\n"
                                 f"{', '.join(long_words[:3])}{'...' if len(long_words) > 3 else ''}")
        
        if not long_words:
            messagebox.showwarning("No Valid Words", 
                                 "No words with 10+ characters found. Please enter longer words or upload again.")
            return
        
        
        threading.Thread(target=self.translate_text, args=(long_words,), daemon=True).start()

    def translate_text(self, long_words):
        """Translate long words in background thread."""
        try:
            translations = {'French': [], 'Hindi': []}
            
            # French
            if 'French' in self.models:
                tokenizer = self.tokenizers['French']
                model = self.models['French']
                
                for word in long_words:
                    try:
                        inputs = tokenizer(word, return_tensors="pt", padding=True)
                        with torch.no_grad():
                            outputs = model.generate(**inputs, max_length=50)
                        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        translations['French'].append(f"{word} → {translated}")
                    except Exception as e:
                        translations['French'].append(f"{word} → [Translation Error]")
            
           
            if 'Hindi' in self.models:
                tokenizer = self.tokenizers['Hindi']
                model = self.models['Hindi']
                
                for word in long_words:
                    try:
                        inputs = tokenizer(word, return_tensors="pt", padding=True)
                        with torch.no_grad():
                            outputs = model.generate(**inputs, max_length=50)
                        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        translations['Hindi'].append(f"{word} → {translated}")
                    except Exception as e:
                        translations['Hindi'].append(f"{word} → [Translation Error]")
            
            self.translate_results['French'] = "\n".join(translations['French'])
            self.translate_results['Hindi'] = "\n".join(translations['Hindi'])
            self.translate_results['Summary'] = (
                f"📊 Translation Summary\n"
                f"• Words Processed: {len(long_words)}\n"
                f"• French Translations: {len(translations['French'])}\n"
                f"• Hindi Translations: {len(translations['Hindi'])}\n\n"
                f"French Translations:\n{self.translate_results['French']}\n\n"
                f"Hindi Translations:\n{self.translate_results['Hindi']}"
            )
            
            self.root.after(0, self.update_results, self.translate_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Translation Error", str(e)))

    def update_results(self, translations):
        self.french_text.delete(1.0, tk.END)
        self.french_text.insert(tk.END, translations.get('French', 'No translations available.\n\n💡 Tip: Only words with 10+ characters are translated.'))
        
        self.hindi_text.delete(1.0, tk.END)
        self.hindi_text.insert(tk.END, translations.get('Hindi', 'No translations available.\n\n💡 Tip: Only words with 10+ characters are translated.'))
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, translations.get('Summary', 'No summary available.'))

    def save_results(self):
        if not any(self.translate_results.values()):
            messagebox.showwarning("No Results", "Nothing to save.")
            return
        
        input_text = self.input_text.get(1.0, tk.END).strip()
        long_words, short_words = self.filter_words_by_length(input_text)
        
        with open('translation_results.txt', 'w', encoding='utf-8') as f:
            f.write("=== DUAL LANGUAGE TRANSLATOR RESULTS ===\n")
            f.write(f"Date: {tk.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if short_words:
                f.write(f"⚠️  SHORT WORDS (need to upload again): {', '.join(short_words)}\n\n")
            
            f.write(f"✅ LONG WORDS PROCESSED ({len(long_words)} words):\n")
            for word in long_words:
                f.write(f"  - {word} ({len(word)} chars)\n")
            f.write("\n")
            
            for lang, result in self.translate_results.items():
                f.write(f"{lang.upper()}:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{result}\n\n")
                
        messagebox.showinfo("Saved", f"Results saved to translation_results.txt\n\n"
                                   f"📝 Processed {len(long_words)} words\n"
                                   f"⚠️  {len(short_words)} short words skipped")

    def copy_all(self):
        input_text = self.input_text.get(1.0, tk.END).strip()
        long_words, short_words = self.filter_words_by_length(input_text)
        
        combined = "=== DUAL LANGUAGE TRANSLATOR RESULTS ===\n\n"
        
        if short_words:
            combined += f"⚠️  SHORT WORDS (<10 chars - upload again):\n"
            combined += f"{', '.join(short_words)}\n\n"
        
        combined += f"✅ LONG WORDS PROCESSED (10+ chars):\n"
        for word in long_words:
            combined += f"• {word} ({len(word)} chars)\n"
        combined += "\n"
        
        combined += "\n".join([f"{lang}: {result}" for lang, result in self.translate_results.items()])
        
        self.root.clipboard_clear()
        self.root.clipboard_append(combined)
        messagebox.showinfo("Copied", f"All results copied to clipboard!\n\n"
                                   f"📝 {len(long_words)} words translated\n"
                                   f"⚠️  {len(short_words)} words need upload")

if __name__ == "__main__":
    root = tk.Tk()
    app = DualLanguageTranslator(root)
    root.mainloop()