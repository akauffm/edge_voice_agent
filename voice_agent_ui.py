# UI based on customtkinter for voice agent
# Default settings meant to be run on tiny screen (Raspberry Pi)

import customtkinter as ctk
import threading
import time
from voice_agent import VoiceAgent, LLmToAudio, AudioToText

class UITextPrinter:
    """Custom printer that updates the UI textbox instead of console"""
    
    def __init__(self, textbox, title=None, clear_on_start=True):
        self.textbox = textbox
        self.title = title
        self.clear_on_start = clear_on_start
        
    def start(self):
        if self.clear_on_start:
            self.textbox.delete("1.0", "end")
        if self.title:
            self.textbox.insert("end", f"--- {self.title} ---\n")
        
        # Add an empty line for partial updates
        self.textbox.insert("end", "\n")
    
    def stop(self):
        self.textbox.insert("end", "\n")
    
    def print(self, transcript, duration=None, partial=False):
        # Always replace the current content of the last line
        last_line_start = self.textbox.index("end-1l linestart")
        self.textbox.delete(last_line_start, "end")
        
        if partial:
            # For partial transcripts, don't add a newline
            self.textbox.insert("end", transcript)
        else:
            # For complete transcripts, add a newline after and prepare a new line for future partials
            self.textbox.insert("end", f"{transcript}\n\n")
        
        self.textbox.see("end")  # Auto-scroll to the end
    
    def show_idle(self, text=None):
        # Make sure we always have a line to update
        if self.textbox.index("end-1c") == "1.0":  # If textbox is empty
            self.textbox.insert("end", "\n")
            
        idle_msg = "..." if text is None else text
        self.print(idle_msg, partial=True)

class VoiceAgentApp:
    def __init__(self, 
                 # UI configuration
                 window_size="480x320",
                 fullscreen=True,
                 label_font_size=14,
                 textbox_font_size=14,
                 button_font_size=16,
                 appearance_mode="dark",
                 color_theme="blue",
                 
                 # Voice agent configuration
                 ollama_model_name="gemma3:1b",
                 tts_engine="piper",
                 asr_model_name="moonshine_onnx_tiny",
                 tts_model_path=None,
                 speaking_rate=1.0,
                 max_words_to_speak_start=5,
                 max_words_to_speak=20,
                 system_prompt=LLmToAudio.DEFAULT_SYSTEM_PROMPT,
                 end_of_utterance_duration=0.5,
                 verbose=False):
                 
        # Voice agent config
        self.ollama_model_name = ollama_model_name
        self.tts_engine = tts_engine
        self.asr_model_name = asr_model_name
        self.tts_model_path = tts_model_path
        self.speaking_rate = speaking_rate
        self.max_words_to_speak_start = max_words_to_speak_start
        self.max_words_to_speak = max_words_to_speak
        self.system_prompt = system_prompt
        self.end_of_utterance_duration = end_of_utterance_duration
        self.verbose = verbose
        
        # UI config
        self.window_size = window_size
        self.fullscreen = fullscreen
        self.label_font_size = label_font_size
        self.textbox_font_size = textbox_font_size
        self.button_font_size = button_font_size
        
        # Initialize the main window
        self.root = ctk.CTk()
        self.root.title("Voice Agent UI")
        
        # Set window size and fullscreen
        self.root.geometry(self.window_size)
        if self.fullscreen:
            self.root.attributes("-fullscreen", True)
        self.root.resizable(False, False)
        
        # Set appearance mode and color theme
        ctk.set_appearance_mode(appearance_mode)
        ctk.set_default_color_theme(color_theme)
        
        # Track running state
        self.is_running = False
        self.agent_thread = None
        
        # Create the UI
        self.create_widgets()
        
        # Bind escape key to exit fullscreen
        self.root.bind("<Escape>", self.exit_fullscreen)
        
        # Initialize VoiceAgent and its components once at startup
        self.voice_agent = VoiceAgent()
        
        # Create custom UI printers for input and output
        self.user_printer = UITextPrinter(self.user_input, title=None, clear_on_start=False)
        self.agent_printer = UITextPrinter(self.agent_output, title=None, clear_on_start=False)
        
        # Initialize the agent components
        self.voice_agent.init_LLmToAudioOutput(
            ollama_model_name=self.ollama_model_name,
            system_prompt=self.system_prompt,
            tts_engine=self.tts_engine,
            speaking_rate=self.speaking_rate,
            tts_model_path=self.tts_model_path,
            max_words_to_speak_start=self.max_words_to_speak_start,
            max_words_to_speak=self.max_words_to_speak,
            verbose=self.verbose,
            printer=self.agent_printer
        )

        self.voice_agent.init_AudioToText(
            asr_model_name=self.asr_model_name,
            end_of_utterance_duration=self.end_of_utterance_duration,
            verbose=self.verbose,
            printer=self.user_printer
        )
        
    def create_widgets(self):
        # Main container frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # User input section
        user_label = ctk.CTkLabel(
            main_frame, 
            text="User Input", 
            font=("Arial", self.label_font_size, "bold")
        )
        user_label.pack(pady=(5, 2))
        
        self.user_input = ctk.CTkTextbox(
            main_frame,
            font=("Arial", self.textbox_font_size),
            wrap="word"
        )
        self.user_input.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        # Agent output section
        agent_label = ctk.CTkLabel(
            main_frame, 
            text="Agent Output", 
            font=("Arial", self.label_font_size, "bold")
        )
        agent_label.pack(pady=(5, 2))
        
        self.agent_output = ctk.CTkTextbox(
            main_frame,
            font=("Arial", self.textbox_font_size),
            wrap="word"
        )
        self.agent_output.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        # Button frame
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=(5, 5))
        
        # Run/Stop button
        self.run_button = ctk.CTkButton(
            button_frame,
            text="Start new agent",
            font=("Arial", self.button_font_size, "bold"),
            height=40,
            command=self.toggle_run
        )
        self.run_button.pack(side="left", padx=(0, 5), fill="x", expand=True)
        
        # Show message context button
        self.context_button = ctk.CTkButton(
            button_frame,
            text="Show Message Context",
            font=("Arial", self.button_font_size, "bold"),
            height=40,
            command=self.show_message_context
        )
        self.context_button.pack(side="left", padx=(0, 5), fill="x", expand=True)
        
        # Exit button
        self.exit_button = ctk.CTkButton(
            button_frame,
            text="Exit",
            font=("Arial", self.button_font_size, "bold"),
            height=40,
            fg_color="#d32f2f",  # Red color for exit button
            hover_color="#b71c1c",  # Darker red on hover
            command=self.exit_application
        )
        self.exit_button.pack(side="right", padx=(5, 0), fill="x", expand=True)
        
    def toggle_run(self):
        if not self.is_running:
            self.start_agent()
        else:
            self.stop_agent()
    
    def start_agent(self):
        if self.is_running:
            return

        if self.agent_thread and self.agent_thread.is_alive():
            return

        self.is_running = True
        self.run_button.configure(text="Terminate agent")
        
        # Clear textboxes and initialize with starting content
        self.user_input.delete("1.0", "end")
        self.user_input.insert("end", "--- User Input --")
        self.agent_output.delete("1.0", "end")
        self.agent_output.insert("end", "--- Assistant Output ---")
        
        # start the voice agent and run in thread
        self.voice_agent.start()      
        self.agent_thread = threading.Thread(target=self.run_agent)
        self.agent_thread.daemon = True
        self.agent_thread.start()
        print('>> Voice agent started')  

    
    def stop_agent(self):
        if self.is_running:
            self.is_running = False
            
            self.agent_output.insert("end", "Stopping...")
            
            # Trigger stop threads early to allow tasks to finish
            self.voice_agent.trigger_stop_events()

            # Then stop the agent
            self.voice_agent.stop()

            # Clear textboxes
            self.user_input.delete("1.0", "end")
            self.user_input.insert("end", "--- terminated ---")
            self.agent_output.delete("1.0", "end")
            self.agent_output.insert("end", "--- terminated ---")
            
            # Try to terminate run thread 
            if self.agent_thread and self.agent_thread.is_alive():
                self.agent_thread.join(timeout=1.0)
            self.agent_thread = None

            self.run_button.configure(text="Start new agent")
    
    def run_agent(self):
        self.voice_agent.run()
    
    def update_ui_after_stop(self):
        self.is_running = False
        self.run_button.configure(text="Start new agent")
    
    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode when Escape is pressed"""
        self.root.attributes("-fullscreen", False)
        self.root.geometry(self.window_size)
    
    def show_message_context(self):
        """Display the current message context in a popup window"""
        if not hasattr(self.voice_agent, 'output_handler') or not hasattr(self.voice_agent.output_handler, 'messages'):
            return
        
        # Get main window size and position
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        
        # Calculate popup size relative to main window (90% of main window size)
        popup_width = int(main_width * 0.9)
        popup_height = int(main_height * 0.9)
        
        # Center the popup relative to the main window
        popup_x = main_x + (main_width - popup_width) // 2
        popup_y = main_y + (main_height - popup_height) // 2
        
        # Create popup window
        popup = ctk.CTkToplevel(self.root)
        popup.title("Message Context")
        popup.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
        popup.transient(self.root)  # Make it transient to main window (will minimize with parent)
        popup.grab_set()  # Make it modal
        
        # Ensure popup comes to foreground and has focus
        popup.lift()  # Lift the window to the top
        popup.focus_force()  # Force focus
        
        # For macOS and some other platforms, additional measures to bring to front
        popup.after(10, lambda: popup.focus_force())  # Force focus again after a short delay
        
        # Create textbox for messages
        message_text = ctk.CTkTextbox(
            popup,
            font=("Arial", self.textbox_font_size),
            wrap="word"
        )
        message_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Format and display messages
        messages = self.voice_agent.output_handler.messages
        formatted_text = ""
        
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            role_display = role.upper()
            formatted_text += f"--- {role_display} ---\n{content}\n\n"
        
        message_text.insert("1.0", formatted_text)
        message_text.see("1.0")  # Scroll to the beginning
        
        # Add close button
        close_button = ctk.CTkButton(
            popup,
            text="Close",
            font=("Arial", self.button_font_size, "bold"),
            command=popup.destroy
        )
        close_button.pack(pady=10)
    
    def exit_application(self):
        """Fully shut down the voice agent and exit the application"""

        # Trigger stop threads early to allow tasks to finish
        self.voice_agent.trigger_stop_events()

        if self.is_running:
            self.stop_agent()
            
        # Properly shutdown resources
        self.voice_agent.shutdown()

        self.root.destroy()
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.root.mainloop()

# Create and run the application
if __name__ == "__main__":
    app = VoiceAgentApp(
        # UI configuration
        window_size="800x800",
        fullscreen=False,
        label_font_size=18,
        textbox_font_size=24,
        button_font_size=20,
    )
    app.run()