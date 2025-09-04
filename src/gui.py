import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from src.summarizer import TextSummarizer
import threading
import math
import traceback

class SummarizerGUI:
    def __init__(self, root):
        print("Initializing SummarizerGUI...")
        self.root = root
        self.root.title("Lecture Summarizer")
        self.root.geometry("1000x800")
        self.is_dark_mode = False
        self.summarizer = TextSummarizer()
        self.font_size = tk.IntVar(value=11)
        self.use_bart = tk.BooleanVar(value=True)
        self.is_summarizing = False
        self.fallback_used = False
        
        self.input_word_count = tk.StringVar(value="Word Count: 0")
        self.summary_word_count = tk.StringVar(value="Word Count: 0")
        self.progress = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Ready")
        self.debug_info = tk.StringVar(value="Ratio: 0.0, Input: 0, Target: 0, Actual: 0, Tokens: 0, T/W: 0.0, Fallback: No")
        
        self.create_widgets()
        print("Widgets created.")
        
        self.input_text.bind("<KeyRelease>", self.update_input_word_count)

    def create_widgets(self):
        print("Creating widgets...")
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", font=("Arial", 11, "bold"))
        self.style.configure("TLabel", font=("Arial", 11))
        self.style.configure("Header.TLabel", font=("Arial", 18, "bold"))
        
        self.main_frame = ttk.Frame(self.root, padding="15")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(
            self.main_frame,
            text="Lecture Summarizer",
            style="Header.TLabel",
            foreground="#2c3e50"
        ).grid(row=0, column=0, columnspan=4, pady=(0, 15))

        input_frame = ttk.LabelFrame(self.main_frame, text="Input Lecture Text", padding="10")
        input_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=10, pady=5)
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame, height=15, wrap=tk.WORD, font=("Arial", self.font_size.get())
        )
        self.input_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        ttk.Label(
            input_frame,
            textvariable=self.input_word_count,
            font=("Arial", 10, "italic")
        ).grid(row=1, column=0, sticky="w", padx=5, pady=2)

        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=10)
        
        ttk.Label(controls_frame, text="Summary Length (Ratio):").grid(row=0, column=0, padx=(0, 10))
        self.ratio_scale = ttk.Scale(
            controls_frame, from_=0.1, to=0.99, orient=tk.HORIZONTAL, command=self.update_ratio_value
        )
        self.ratio_scale.set(0.3)
        self.ratio_scale.grid(row=0, column=1, padx=5)
        
        self.ratio_value = tk.StringVar(value="0.3")
        ttk.Label(controls_frame, textvariable=self.ratio_value).grid(row=0, column=2, padx=5)
        
        ttk.Checkbutton(
            controls_frame, text="Use BART (uncheck for TextRank)", variable=self.use_bart
        ).grid(row=0, column=3, padx=(20, 10))
        
        ttk.Label(controls_frame, text="Font Size:").grid(row=0, column=4, padx=(10, 10))
        ttk.Scale(
            controls_frame, from_=10, to=16, orient=tk.HORIZONTAL, variable=self.font_size,
            command=self.update_font_size
        ).grid(row=0, column=5, padx=5)
        
        ttk.Button(
            controls_frame, text="Toggle Theme", command=self.toggle_theme
        ).grid(row=0, column=6, padx=(10, 5))

        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=10)
        
        self.summarize_button = ttk.Button(
            button_frame, text="Summarize", command=self.start_summarize, style="Accent.TButton"
        )
        self.summarize_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(
            button_frame, text="Clear", command=self.clear_text
        ).grid(row=0, column=1, padx=5)
        
        self.progress_bar = ttk.Progressbar(
            button_frame, variable=self.progress, maximum=100
        ).grid(row=0, column=2, sticky="ew", padx=5)
        button_frame.columnconfigure(2, weight=1)
        
        ttk.Label(
            button_frame, textvariable=self.status, font=("Arial", 10)
        ).grid(row=0, column=3, padx=5)
        
        ttk.Label(
            button_frame, textvariable=self.debug_info, font=("Arial", 10)
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=5, pady=2)

        output_frame = ttk.LabelFrame(self.main_frame, text="Summary Output", padding="10")
        output_frame.grid(row=4, column=0, columnspan=4, sticky="nsew", padx=10, pady=5)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame, height=10, wrap=tk.WORD, font=("Arial", self.font_size.get())
        )
        self.output_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        ttk.Label(
            output_frame,
            textvariable=self.summary_word_count,
            font=("Arial", 10, "italic")
        ).grid(row=1, column=0, sticky="w", padx=5, pady=2)

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(4, weight=1)

    def update_input_word_count(self, event=None):
        text = self.input_text.get("1.0", tk.END).strip()
        word_count = len(text.split())
        self.input_word_count.set(f"Word Count: {word_count}")

    def update_ratio_value(self, value):
        self.ratio_value.set(f"{float(value):.2f}")

    def update_font_size(self, value):
        font_size = int(float(value))
        self.input_text.configure(font=("Arial", font_size))
        self.output_text.configure(font=("Arial", font_size))

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        if self.is_dark_mode:
            self.root.configure(bg="#2c3e50")
            self.style.configure("Header.TLabel", foreground="#ecf0f1")
            self.style.configure("TLabel", foreground="#ecf0f1", background="#2c3e50")
            self.style.configure("Accent.TButton", background="#3498db")
        else:
            self.root.configure(bg="#f0f2f5")
            self.style.configure("Header.TLabel", foreground="#2c3e50")
            self.style.configure("TLabel", foreground="#2c3e50", background="#f0f2f5")
            self.style.configure("Accent.TButton", background="#3498db")

    def start_summarize(self):
        if self.is_summarizing:
            return
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "Please enter text to summarize.")
            return
        self.is_summarizing = True
        self.summarize_button.configure(state="disabled")
        self.status.set("Summarizing...")
        self.progress.set(0)
        self.fallback_used = False
        threading.Thread(target=self.summarize_text, daemon=True).start()

    def summarize_text(self):
        try:
            input_text = self.input_text.get("1.0", tk.END).strip()
            ratio = self.ratio_scale.get()
            use_bart = self.use_bart.get()
            input_words = len(input_text.split())
            target_words = math.ceil(input_words * ratio)
            
            def progress_callback(value):
                self.progress.set(value)
                self.root.update()
            
            progress_callback(30)
            summary = self.summarizer.summarize(input_text, ratio, use_bart=use_bart, progress_callback=progress_callback)
            progress_callback(80)
            
            self.root.after(0, lambda: self.output_text.delete("1.0", tk.END))
            self.root.after(0, lambda: self.output_text.insert(tk.END, summary))
            word_count = len(summary.split())
            output_tokens = len(self.summarizer.tokenizer.encode(summary)) if self.summarizer.use_bart and word_count > 0 else 0
            token_ratio = output_tokens / word_count if word_count > 0 else 0
            fallback_status = "Yes" if self.fallback_used or (use_bart and word_count < target_words * 0.9) else "No"
            self.root.after(0, lambda: self.summary_word_count.set(f"Word Count: {word_count}"))
            self.root.after(0, lambda: self.debug_info.set(f"Ratio: {ratio:.2f}, Input: {input_words}, Target: {target_words}, Actual: {word_count}, Tokens: {output_tokens}, T/W: {token_ratio:.2f}, Fallback: {fallback_status}"))
            progress_callback(100)
            self.root.after(0, lambda: self.status.set("Ready"))
            self.root.after(0, lambda: self.summarize_button.configure(state="normal"))
            self.root.after(0, lambda: setattr(self, "is_summarizing", False))
            self.root.after(0, lambda: self.progress.set(0))
        except Exception as e:
            print(f"Error in summarize_text: {str(e)}\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
            self.root.after(0, lambda: self.status.set("Error"))
            self.root.after(0, lambda: self.summarize_button.configure(state="normal"))
            self.root.after(0, lambda: setattr(self, "is_summarizing", False))
            self.root.after(0, lambda: self.progress.set(0))
            self.root.after(0, lambda: self.debug_info.set("Ratio: 0.0, Input: 0, Target: 0, Actual: 0, Tokens: 0, T/W: 0.0, Fallback: No"))

    def clear_text(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.input_word_count.set("Word Count: 0")
        self.summary_word_count.set("Word Count: 0")
        self.progress.set(0)
        self.status.set("Ready")
        self.debug_info.set("Ratio: 0.0, Input: 0, Target: 0, Actual: 0, Tokens: 0, T/W: 0.0, Fallback: No")