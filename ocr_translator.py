import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import easyocr
import langdetect
from langdetect import DetectorFactory
from googletrans import Translator, LANGUAGES  # FIXED: Added LANGUAGES
import threading
import os
from pathlib import Path

DetectorFactory.seed = 0

class OCRTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Text Extractor & Translator")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        try:
            self.ocr_reader = easyocr.Reader(['en'])  # English only
        except Exception as e:
            messagebox.showerror("OCR Error", f"Failed to initialize OCR: {e}\nMake sure EasyOCR is installed correctly.")
            self.ocr_reader = None
            return
            
        self.translator = Translator()
        self.supported_languages = {
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Dutch': 'nl',
            'Russian': 'ru',
            'Chinese': 'zh-cn',
            'Japanese': 'ja',
            'Korean': 'ko'
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        title_label = ttk.Label(main_frame, text="OCR Text Extractor & Translator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Button(input_frame, text="📁 Upload Image/Video", 
                  command=self.upload_file).grid(row=0, column=0, padx=(0, 10), pady=5)
        
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(input_frame, textvariable=self.file_path_var, 
                                       state='readonly', width=50)
        self.file_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=5)
        
        self.preview_label = ttk.Label(input_frame, text="Preview will appear here")
        self.preview_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(1, weight=1)
        
        ttk.Label(options_frame, text="Translate to:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.target_lang_var = tk.StringVar(value='Spanish')
        target_combo = ttk.Combobox(options_frame, textvariable=self.target_lang_var, 
                                   values=list(self.supported_languages.keys()), 
                                   state='readonly', width=15)
        target_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(options_frame, text="Mode:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.mode_var = tk.StringVar(value='Image')
        mode_combo = ttk.Combobox(options_frame, textvariable=self.mode_var,
                                 values=['Image', 'Video'], state='readonly', width=10)
        mode_combo.grid(row=0, column=3, sticky=tk.W)
        
        self.process_btn = ttk.Button(options_frame, text="🔍 Process & Translate", 
                                     command=self.start_processing)
        self.process_btn.grid(row=0, column=4, padx=(20, 0))
        
        self.progress = ttk.Progressbar(options_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
        
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(output_frame, height=15, width=80, 
                                                    wrap=tk.WORD, font=('Consolas', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        stats_frame = ttk.Frame(output_frame)
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.stats_label = ttk.Label(stats_frame, text="", foreground='blue')
        self.stats_label.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(stats_frame, text="💾 Save Results", 
                  command=self.save_results).grid(row=0, column=1, padx=(20, 0))
        
    def upload_file(self):
        file_types = {
            "Image files": ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"],
            "Video files": ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv"],
            "All files": ["*.*"]
        }
        
        file_path = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("All supported", "*.png *.jpg *.jpeg *.bmp *.tiff *.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.display_preview(file_path)
    
    def display_preview(self, file_path):
        try:
            for widget in self.preview_label.winfo_children():
                widget.destroy()
            
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img = Image.open(file_path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                preview_img = ttk.Label(self.preview_label, image=photo)
                preview_img.image = photo  # Keep a reference
                preview_img.grid(row=0, column=0)
                
                ttk.Label(self.preview_label, text=f"Image: {os.path.basename(file_path)}").grid(row=1, column=0)
                
            else:
                cap = cv2.VideoCapture(file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                
                ttk.Label(self.preview_label, 
                         text=f"Video: {os.path.basename(file_path)}\n"
                              f"Duration: {duration:.1f}s | FPS: {fps:.1f} | Frames: {frame_count:.0f}").grid(row=0, column=0)
                cap.release()
                
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not display preview: {str(e)}")
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using EasyOCR"""
        if not self.ocr_reader:
            return [], "OCR not initialized"
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                return [], "Could not read image"
            
            processed = self.preprocess_image(image)
            
            results = self.ocr_reader.readtext(processed, detail=1, paragraph=False)
            
            extracted_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Confidence threshold
                    extracted_texts.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox
                    })
            
            return extracted_texts, None
            
        except Exception as e:
            return [], f"OCR Error: {str(e)}"
    
    def extract_text_from_video(self, video_path):
        """Extract text from video frames"""
        if not self.ocr_reader:
            return [], "OCR not initialized"
            
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return [], "Could not open video"
            
            extracted_texts = []
            frame_count = 0
            processed_frames = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_interval = 30
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue
                
                processed_frame = self.preprocess_image(frame)
                
                results = self.ocr_reader.readtext(processed_frame, detail=1, paragraph=False)
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.6:
                        extracted_texts.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': bbox,
                            'frame': frame_count
                        })
                
                processed_frames += 1
                if processed_frames % 10 == 0:
                    self.root.after(0, lambda: self.progress['value'] + 1)
            
            cap.release()
            return extracted_texts, None
            
        except Exception as e:
            return [], f"Video OCR Error: {str(e)}"
    
    def detect_language(self, text):
        """Detect if text is English"""
        try:
            if len(text) < 3:
                return 'unknown'
            
            detected = langdetect.detect(text)
            return detected if detected == 'en' else 'non-english'
            
        except:
            return 'unknown'
    
    def translate_text(self, text, target_lang):
        """Translate English text to target language"""
        try:
            if not text or len(text.strip()) < 2:
                return text
            
            target_code = self.supported_languages.get(target_lang, 'es')
            
            # Detect if English first
            lang = self.detect_language(text)
            if lang != 'en':
                return f"[Non-English detected: {lang}] {text}"
            
            # Translate
            translated = self.translator.translate(text, src='en', dest=target_code)
            return translated.text
            
        except Exception as e:
            return f"[Translation Error: {str(e)}] {text}"
    
    def start_processing(self):
        """Start processing in a separate thread"""
        if not self.file_path_var.get():
            messagebox.showwarning("No File", "Please select a file first!")
            return
        
        if not self.ocr_reader:
            messagebox.showerror("OCR Error", "OCR model not loaded. Please restart the application.")
            return
            
        # Start progress bar
        self.progress.start()
        self.process_btn.config(state='disabled')
        self.results_text.delete(1.0, tk.END)
        self.stats_label.config(text="")
        
        # Start processing thread
        thread = threading.Thread(target=self.process_file)
        thread.daemon = True
        thread.start()
    
    def process_file(self):
        """Process the selected file"""
        file_path = self.file_path_var.get()
        target_lang = self.target_lang_var.get()
        mode = self.mode_var.get()
        
        try:
            if mode == 'Image':
                texts, error = self.extract_text_from_image(file_path)
            else:
                texts, error = self.extract_text_from_video(file_path)
            
            if error:
                self.root.after(0, lambda: self.show_error(error))
                return
            
            # Process results
            english_texts = []
            translations = []
            
            for item in texts:
                original_text = item['text']
                confidence = item['confidence']
                
                lang = self.detect_language(original_text)
                if lang == 'en':
                    translated_text = self.translate_text(original_text, target_lang)
                    english_texts.append(original_text)
                    translations.append(translated_text)
                    
                    item['original'] = original_text
                    item['translated'] = translated_text
                    item['language'] = lang
                else:
                    item['original'] = original_text
                    item['translated'] = f"[Non-English: {lang}] {original_text}"
                    item['language'] = lang
            
            self.root.after(0, lambda: self.display_results(texts, english_texts, translations, target_lang))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Processing Error: {str(e)}"))
        
        finally:
            # Stop progress bar and re-enable button
            self.root.after(0, self.processing_complete)
    
    def display_results(self, texts, english_texts, translations, target_lang):
        """Display results in GUI"""
        self.results_text.delete(1.0, tk.END)
        
        if not texts:
            self.results_text.insert(tk.END, "No text detected in the image/video.\n\n"
                                           "Tips:\n"
                                           "• Ensure text is clear and well-lit\n"
                                           "• Try adjusting image contrast\n"
                                           "• For videos, ensure text is stable")
            return
        
        self.results_text.insert(tk.END, f"=== OCR & Translation Results ===\n")
        self.results_text.insert(tk.END, f"Target Language: {target_lang}\n")
        self.results_text.insert(tk.END, f"Total Text Blocks: {len(texts)}\n")
        self.results_text.insert(tk.END, f"English Text Found: {len(english_texts)}\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n\n")
        
        for i, item in enumerate(texts, 1):
            original = item['original']
            translated = item['translated']
            confidence = item['confidence']
            lang = item.get('language', 'unknown')
            
            self.results_text.insert(tk.END, f"[{i}] Confidence: {confidence:.1%}\n")
            self.results_text.insert(tk.END, f"Language: {lang}\n")
            self.results_text.insert(tk.END, f"Original: {original}\n")
            
            if lang == 'en' and original != translated:
                self.results_text.insert(tk.END, f"Translated: {translated}\n")
            
            self.results_text.insert(tk.END, "-" * 30 + "\n\n")
        
        if english_texts:
            self.results_text.insert(tk.END, "\n=== ENGLISH TEXT SUMMARY ===\n")
            for i, (orig, trans) in enumerate(zip(english_texts, translations), 1):
                self.results_text.insert(tk.END, f"{i}. {orig} → {trans}\n")
        
        stats = f"📊 Stats: {len(texts)} blocks | {len(english_texts)} English | {len(english_texts)/max(len(texts),1)*100:.0f}% success"
        self.stats_label.config(text=stats)
    
    def show_error(self, error_msg):
        """Show error message"""
        messagebox.showerror("Processing Error", error_msg)
        self.results_text.insert(tk.END, f"\nError: {error_msg}\n")
    
    def processing_complete(self):
        """Called when processing is complete"""
        self.progress.stop()
        self.process_btn.config(state='normal')
    
    def save_results(self):
        """Save results to file"""
        if not self.results_text.get(1.0, tk.END).strip():
            messagebox.showwarning("No Results", "No results to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save results as..."
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save file: {str(e)}")

def main():
    root = tk.Tk()
    app = OCRTranslatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()