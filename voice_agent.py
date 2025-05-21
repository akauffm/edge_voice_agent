# Fully offline running voice agent.
#
# Uses Ollama for on-devive LLMs. Supports several on-device runnable tts-engines and asr models.
# Defaults are set for smallest models so that it can run on edge devices like Raspberry Pi 5.
#
# Example for an outdoor survival guide:
# 
# python voice_agent.py \
#   --system_prompt "You are an outdoor survival guide assistant helping users, who have no internet access, no phone access, and are far from civilization to deal with challenges they experience in the outdoors. Please give helpful advice, but be VERY brief. Only give details when asked." \
#   --speaking_rate 3.0


from captioning_lib import captioning_utils
from captioning_lib import printers
import random
import pyaudio
import queue
import json
import ollama
import sounddevice as sd
import nltk
from nltk.tokenize import sent_tokenize
import time
import threading
import argparse
import queue
import signal
import sys
from tts_lib import tts_engines
import time 
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
import re
import emoji


DEFAULT_SYSTEM_PROMPT = """You are an assistant that runs on an edge device. Respond with a single, short sentence only."""

# DEFAULT_SYSTEM_PROMPT = """You are an assistant that runs on an edge device. A person is interacting with you via voice. 
# For that reason you should limit your answers a bit in length unless explicitly asked to give detailed responses. 
# If you are asked for advise, list all relevant points but limit yourself to the top 3 items.
# But most importantly, do not output any sort of formatting information.
# Do not start your sentences with 'okay' always. Be friendly and helpful."""

MAX_TEXT_BUFFER = 125
MIN_TEXT_BUFFER = 75

class ColoredPrinter(printers.CaptionPrinter):

    def __init__(self, title, title_color='blue'):
        # https://rich.readthedocs.io/en/stable/style.html
        from rich.console import Console
        from rich.theme import Theme
        self.title = title
        self.title_color = title_color
        caption_theme = Theme({
            "partial": "italic",
            "segment": f"bold {self.title_color}",
        })
        self.console = Console(theme=caption_theme, highlight=False)

    def start(self):
        self.console.rule(f"[bold {self.title_color}]{self.title}")

    def stop(self):
        self.console.rule()

    def print(self, transcript, duration=None, partial=False):
        """Update the caption display with the latest transcription"""
        # Move to the beginning of the line and clear it
        sys.stdout.write("\r\033[K")  

        text = transcript

        # Show partial and full segments differently
        if partial:
            terminal_width = self.console.width
            if len(text) > terminal_width/2:
                last_chars = terminal_width - 5
                text = '...' + text[-last_chars:]
            syle = "partial"
            self.console.print(text, end="", style=syle)   # Print the styled text without adding a new line
        else:
            syle = "segment"
            self.console.print(text, style=syle) # Print the styled text without adding a new line


class OllamaToPiperStreamer:
    def __init__(self, 
                 ollama_model_name="gemma3:1b", 
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 tts_engine='piper',
                 speaking_rate=1.0, # higher numbers means faster
                 tts_model_path=None,
                 verbose=False
                 ):
        """Initialize the streamer with Piper and Ollama models."""
        self.verbose = verbose

        self.ollama_model_name = ollama_model_name
        print(f"Using Ollama model: {self.ollama_model_name}")

        self.ollama_options={
            "temperature": 0.3,  # lower temperature for speed
            "num_predict": -1,  # unlimited
            #"num_ctx": 1024
        }
        # warm up model
        t1 = time.time()
        ollama.chat(
            model=self.ollama_model_name,
            messages=[{"role": "user", "content": "hi"}],
            stream=False
        )        
        self._info(f"Ollama model warmed up in {time.time()-t1} secs.")

        # initialize conversation context
        self.messages = [
            {'role': 'system', 'content': system_prompt},
        ]

        # load TTS
        if tts_engine == 'piper':
            print('Initializing Piper TTS')
            if tts_model_path:
                print(f"Initializing with tts model: {tts_model_path}")                
                self.tts = tts_engines.TTS_Piper(tts_model_path)
            else:
                self.tts = tts_engines.TTS_Piper()
        elif tts_engine == 'kokoro':
            print('Initializing Kokoro TTS')
            if tts_model_path:
                print(f"Initializing with tts model: {tts_model_path}")                
                self.tts = tts_engines.TTS_Kokoro(tts_model_path)
            else:
                self.tts = tts_engines.TTS_Kokoro()
        else:
            raise ValueError('Unknown tts engine.')
        
        self.sample_rate = self.tts.get_sample_rate()
        print(f"Using sample rate: {self.sample_rate} Hz")
        
        self.speaking_rate = speaking_rate
        print(f"Using speaking rate: {self.speaking_rate}")        

        # Initialize audio stream
        self.audio_stream = None
        
        # Text processing
        self.text_buffer = ""
        self.sentence_queue = queue.Queue()
        self.is_processing = False
        self.is_speaking = False
        self.lock = threading.Lock()
        
        # Create stop event for clean termination
        self.stop_event = threading.Event()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # printer for output
        self.assistant_printer = ColoredPrinter("Assistant Output", "magenta")
    
        # keyboard interruption

    def _clean_llm_output(self, text):
        """
        Remove formatting symbols.
        """
        text = text.replace('*', ' ')
        text = emoji.replace_emoji(text, replace='')
        return text
            
    def _info(self, text):
        if self.verbose:
            print(text)

    def _signal_handler(self, sig, frame):
        """Handle termination signals gracefully."""
        self._info("\nReceived termination signal. Shutting down...")
        self.stop_event.set()
        self._close()
        sys.exit(0)
    
    def _start_audio_stream(self):
        """Initialize and start the audio output stream."""

        # increase buffer size if needed, esp on slower devices like raspberry pi
        buffer_size = 1024

        if self.audio_stream is None:
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=buffer_size,
                channels=1,
                dtype='int16'
            )
            self.audio_stream.start()
            self._info("Audio stream started")
    
    def _start_sentence_processor(self):
        """Start a background thread to process sentences."""
        if self.is_processing:
            return
            
        self.is_processing = True
        threading.Thread(target=self._process_sentences, daemon=True).start()
    
    def _close(self):
        """Close all resources."""
        if self.audio_stream:
            self._info("Closing audio stream...")
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
    
    def _process_text_chunk(self, text_chunk):
        """Process a chunk of text from Ollama, detecting complete sentences."""
        if self.stop_event.is_set():
            return
            
        if not text_chunk:
            return

        with self.lock:
            self.text_buffer += text_chunk
            
            # find complete sentences
            try:
                sentences = sent_tokenize(self.text_buffer)
                if len(sentences) > 1:
                    complete_sentences = sentences[:-1]
                    
                    # Keep the last (potentially incomplete) sentence in buffer
                    self.text_buffer = sentences[-1]
                    
                    # Add complete sentences to the queue
                    for sentence in complete_sentences:
                        if sentence.strip():
                            self.sentence_queue.put(sentence)
                            self._info(f"Queued: {sentence}")
                
                # If buffer is getting long but no sentence breaks, force process
                elif len(self.text_buffer) > MAX_TEXT_BUFFER:
                    # Look for natural break points
                    break_points = [
                        self.text_buffer.rfind(', '),
                        self.text_buffer.rfind(' - '),
                        self.text_buffer.rfind(': '),
                        self.text_buffer.rfind(' ')
                    ]
                    
                    # Find the best break point
                    break_point = max(break_points)
                    
                    if break_point > MIN_TEXT_BUFFER:  # Ensure we're not breaking too early
                        fragment = self.text_buffer[:break_point+1]
                        self.text_buffer = self.text_buffer[break_point+1:]
                        self.sentence_queue.put(fragment)
                        self._info(f"Forced break: {fragment}")
            
            except Exception as e:
                print(f"Error in sentence detection: {e}")
        
        # Ensure the sentence processor is running
        if not self.is_processing:
            self._start_sentence_processor()
    
    def _process_sentences(self):
        """Process sentences from the queue and speak them."""
        self._start_audio_stream()
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get a sentence from the queue with timeout
                    sentence = self.sentence_queue.get(timeout=0.5)
                    
                    # Wait until not speaking to avoid overlap
                    while self.is_speaking and not self.stop_event.is_set():
                        time.sleep(0.05)
                    
                    if self.stop_event.is_set():
                        break
                    
                    # Speak the sentence
                    self._speak_sentence(sentence, speed=self.speaking_rate)
                    self.sentence_queue.task_done()
                
                except queue.Empty:
                    # No sentences to process
                    if self.sentence_queue.empty() and not self.text_buffer and not self.is_speaking:
                        # Exit if we've processed everything
                        break
        
        finally:
            self.is_processing = False
            
            # If there are still sentences and we're not stopped, restart processor
            if not self.sentence_queue.empty() and not self.stop_event.is_set():
                self._start_sentence_processor()
    
    def _speak_sentence(self, text, speed=1.0, noise_scale=0.667, noise_w=0.8):
        """Synthesize and play a sentence with Piper."""
        if not text.strip():
            return

        self.is_speaking = True
        self._info(f"Speaking: {text}")
        
        try:
            audio_data, sample_rate = self.tts.synthesize(
                text, target_sr = self.sample_rate, 
                speaking_rate=speed, return_as_int16=True)
            self.audio_stream.write(audio_data)

        except Exception as e:
            print(f"Error synthesizing speech: {e}")
        
        finally:
            self.is_speaking = False
    
    def _finish_processing(self):
        """Process any remaining text in the buffer."""
        with self.lock:
            if self.text_buffer.strip():
                self.sentence_queue.put(self.text_buffer)
                self.text_buffer = ""
        
        # Wait for sentence queue to empty
        if self.sentence_queue.qsize() > 0:
            self._info(f"Waiting for {self.sentence_queue.qsize()} pending sentences...")
            self.sentence_queue.join()
        
        # Give a moment for audio to finish playing
        time.sleep(0.5)
    
    def process_prompt(self, user_prompt):
        """Process a prompt through Ollama and stream to Piper."""

        self.assistant_printer.start()

        self.messages.append({'role': 'user', 'content': user_prompt})

        self._info(f">> context length: turns: {len(self.messages) / 2}")
        self._info(f">> context length: characters: {len(json.dumps(self.messages))}")

        pretty_json = json.dumps(self.messages, indent=2)
        self._info(f">> Sending prompt to {self.ollama_model_name}: {pretty_json}")

        response = ollama.chat(
            model=self.ollama_model_name,
            messages=self.messages,
            stream=True,
            options=self.ollama_options
        )
        

        text_chunks = []
        for chunk in response:

            if self.stop_event.is_set():
                break
                
            if chunk and 'message' in chunk and 'content' in chunk['message']:
                text_chunk = chunk['message']['content']


                # remove asterisks and other formatting info from the text
                text_chunk = self._clean_llm_output(text_chunk)

                self._process_text_chunk(text_chunk)            
                text_chunks.append(text_chunk)

                # show some activity until all output is produced
                spinner_symbol = random.choice(['+', 'o'])
                self.assistant_printer.print(spinner_symbol, partial=True)

        assistant_response = ''.join(text_chunks)
        self.messages.append({'role': 'assistant', 'content': assistant_response})

        # print final assistant message
        self.assistant_printer.print(assistant_response, partial=False)

        # Process any remaining text
        self._finish_processing()
            

class UserInputFromAudio:
    def __init__(self, asr_model_name,
                 end_of_utterance_duration=0.5, 
                 min_partial_duration=0.2, max_segment_duration=15,
                 verbose=False):

        self.verbose = verbose
        self.end_of_utterance_duration = end_of_utterance_duration
        
        self.min_partial_duration = min_partial_duration
        self.max_segment_duration = max_segment_duration
        
        self.caption_printer = ColoredPrinter("User Input", "blue")
        
        self.audio_queue = queue.Queue(maxsize=1000)
        self.vad = captioning_utils.get_vad(eos_min_silence=200)    
        self.asr_model = captioning_utils.load_asr_model(asr_model_name, 16000, False)

        # Start transcription thread
        self.stop_threads = threading.Event()  # Event to signal threads to stop    

        self.transcription_handler = captioning_utils.TranscriptionWorker(sampling_rate=captioning_utils.SAMPLING_RATE)


        self.transcriber = threading.Thread(target=self.transcription_handler.transcription_worker, 
                                    kwargs={'vad': self.vad,
                                            'asr': self.asr_model,
                                            'audio_queue': self.audio_queue,
                                            'caption_printer': self.caption_printer,
                                            'stop_threads': self.stop_threads,
                                            'min_partial_duration': self.min_partial_duration,
                                            'max_segment_duration': self.max_segment_duration})
        self._info(f">>>handler started, recording: {self.transcription_handler.is_speech_recording}")
        self.transcriber.daemon = True
        self.transcriber.start()

        audio = pyaudio.PyAudio()
        input_device = captioning_utils.find_default_input_device()
        self._info(f"Using default audio input device: {input_device}")
        device_index = input_device['index']
        self.input_audio_stream = captioning_utils.get_audio_stream(audio, input_device_index=device_index)

    def shutdown(self):
        time.sleep(0.2)
        self.stop_threads.set()

        # Empty queue
        while not self.audio_queue.empty():
            logging.debug("Emptying audio queue...")
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break

        # Clean up audio resources
        if self.input_audio_stream:
            self.input_audio_stream.stop_stream()
            self.input_audio_stream.close()

    def _info(self, text):
        if self.verbose:
            print(text)

    def get_speech_input(self):

        self.caption_printer.start()

        self.input_audio_stream.start_stream()
        self._info(f">>> START input audio stream active: {self.input_audio_stream.is_active()}")

        while True:
            data = self.input_audio_stream.read(captioning_utils.AUDIO_FRAMES_TO_CAPTURE)
            self.audio_queue.put(data, timeout=0.1)

            all_transcribed = ''
            if not self.transcription_handler.is_speech_recording:
                if not self.transcription_handler.had_speech:
                    # wait until user spoke
                    self.caption_printer.print("please speak", partial=True)
                else:
                    # define EOU when we haven't seen speech for a while
                    if self.transcription_handler.time_since_last_speech() > self.end_of_utterance_duration:
                        self._info(">>> seems user stopped speaker...")
                        # retranscribe and capture all
                        all_transcribed = ' '.join(self.transcription_handler.transcribed_segments)
                        self._info(f">> all said: {all_transcribed}")
                        # reset the transcriber
                        self.transcription_handler.reset()
                        break
        self._info('>>> --- finished speaking')
        self.input_audio_stream.stop_stream()
        self._info(f">>> END input audio stream active: {self.input_audio_stream.is_active()}")

        return all_transcribed


def main():
    """Main function to run the Ollama to Piper streamer."""
    parser = argparse.ArgumentParser(description="Stream text from Ollama to Piper TTS")
    parser.add_argument("--ollama-model-name", default="gemma3:1b", help="Ollama model to use (default: llama3)")
    parser.add_argument("--tts_engine", choices=['piper', 'kokoro'], default="piper", help="which tts engine to use; piper is much faster than kokoro.")
    parser.add_argument("--asr_model_name", default="moonshine_onnx_tiny", help="which asr model to run.")
    parser.add_argument("--tts_model_path", required=False, help="Path to the tts model (.onnx file)")
    parser.add_argument("--speaking_rate", type=float, default=1.0, help="how fast should generated speech be, 1.0 is default, higher numbers mean faster speech")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT, help="Instructions for the model.")
    parser.add_argument("--end_of_utterance_duration", type=float, default=0.5, help="Silence seconds until end of turn of user identified")

    parser.add_argument("--verbose", action="store_true", help="Verbose status info")
    
    args = parser.parse_args()
    

    streamer = OllamaToPiperStreamer(
        ollama_model_name=args.ollama_model_name,
        system_prompt=args.system_prompt,
        tts_engine=args.tts_engine,
        speaking_rate=args.speaking_rate,
        tts_model_path=args.tts_model_path,
        verbose=args.verbose
    )

    user_input_reader = UserInputFromAudio(
        args.asr_model_name,
        end_of_utterance_duration=0.5, verbose=args.verbose)

    
    try:
            
        while True:
            user_input_transcribed = user_input_reader.get_speech_input()

            prompt = user_input_transcribed
            if 'please exit' in prompt.lower():
                streamer._speak_sentence('Goodbye, then!')
                break
            
            if prompt:
                streamer.process_prompt(prompt)

            #streamer.user_interrupt_event = None
            
    finally:
        user_input_reader.shutdown()
        streamer._close()



if __name__ == "__main__":
    main()
