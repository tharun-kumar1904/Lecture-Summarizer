import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tkinter as tk
from src.gui import SummarizerGUI

def main():
    print("Starting main function...")
    root = tk.Tk()
    print("Tk root created.")
    app = SummarizerGUI(root)
    print("SummarizerGUI initialized.")
    root.mainloop()
    print("Mainloop exited.")

if __name__ == "__main__":
    print("Running main.py...")
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        raise