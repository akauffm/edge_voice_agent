# Fully offline running voice agent.
#
# Uses Ollama for on-devive LLMs. Supports several on-device runnable tts-engines and asr models.
# Defaults are set for smallest models so that it can run on edge devices like Raspberry Pi 5.


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

    def show_idle(self, text=None):
        spinner_symbol = random.choice(['+', 'o'])
        idle_msg = spinner_symbol
        if text:
            idle_msg = text
        self.print(idle_msg, partial=True)


class LLmToAudio:
    """Generate LLM output based on prompt and stream into TTS output."""

    DEFAULT_SYSTEM_PROMPT = """You are an assistant that runs on an edge device. Respond with a single, short sentence only."""

    def __init__(self, 
                 ollama_model_name="gemma3:1b", 
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 tts_engine='piper',
                 speaking_rate=1.0, # higher numbers means faster
                 tts_model_path=None,
                 max_words_to_speak_start=5,  # make sure that we get to speak quickly at the beginning
                 max_words_to_speak=15, # later speak at last after this many words, or when a sentence is finished
                 verbose=False,
                 printer=None
                 ):
        """Initialize the streamer with Piper and Ollama models."""
        self.verbose = verbose

        ## Init LLM
        self.system_prompt = system_prompt
        self.messages = [
            {'role': 'system', 'content': self.system_prompt},
        ]

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

        # Init TTS
        self.max_words_to_speak_start = max_words_to_speak_start
        self.max_words_to_speak = max_words_to_speak
        assert self.max_words_to_speak_start <= self.max_words_to_speak

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

        # increase buffer size if needed, esp on slower devices like raspberry pi
        self.audio_buffer_size = 2048

        # Printer
        if not printer:
            self.assistant_printer = ColoredPrinter("Agent Output", "magenta")
        else:
            self.assistant_printer = printer

        self.audio_stream = None

        # Thread handlers
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # Signal handlers for graceful termination
        if threading.current_thread() is threading.main_thread():
            print('>>> setting up signal handlers')
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    
    def start(self):
        """Start and restart the agent."""
        # Reset LLM context
        self.messages = [
            {'role': 'system', 'content': self.system_prompt},
        ]

        # Text processing
        self.text_buffer = ""
        self.sentence_queue = queue.Queue()
        self.is_processing = False
        self.is_speaking = False

        # marker for first words spoken
        self.first_speech_fragment_finalized = False
        self.time_llm_gen_started = time.time()
        self.first_chunk_emitted = False


    def stop(self):
        """Stop agent."""
        self.stop_event.clear()

        # Reset text buffers and queues
        while not self.sentence_queue.empty():
            self.sentence_queue.get_nowait()
            self.sentence_queue.task_done()

        # Clear states
        self.messages = []
        self.text_buffer = ""


    def shutdown(self):
        """Close all resources."""
        if self.audio_stream:
            self._info("Closing audio stream...")
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None


    def _clean_llm_output(self, text):
        """
        Remove formatting symbols we don't want to be spoken.
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
        self.shutdown()
        sys.exit(0)
    
    def _start_audio_stream(self):
        """Initialize and start the audio output stream."""

        if self.audio_stream is None:
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.audio_buffer_size,
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
        
    def _get_max_buffer_words_before_speaking(self):
        # If unspoken text buffer is getting long until first sentence break observed, we will need to
        # create a artificial break to ensure latency doesn't get too big.
        # This is more critical at the beginning of a response, before we have started speaking, where
        # the goal is to minimize time to speech onset.
        if not self.first_speech_fragment_finalized:
            return self.max_words_to_speak_start
        else:
            return self.max_words_to_speak

    def _process_text_chunk(self, text_chunk):
        """Process a chunk of text from Ollama.
        
        Decide when to put in the speak queue based on sentence end detection on max chunk size."""
        if self.stop_event.is_set():
            return
            
        if not text_chunk:
            return

        with self.lock:
            self.text_buffer += text_chunk
            self.text_buffer_words = self.text_buffer.split()
            
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
                            self._info(f"Queued full sentence: {sentence}")
                            if not self.first_speech_fragment_finalized:
                                self.time_to_first_speech_fragment = time.time() - self.time_llm_gen_started 
                                self._info(f"\n>> Time to first speech fragment (organic): {self.time_to_first_speech_fragment:.2f} seconds")
                            self.first_speech_fragment_finalized = True

                elif len(self.text_buffer_words) > self._get_max_buffer_words_before_speaking():
                    # Look for natural break points
                    break_points = [
                        self.text_buffer.rfind(', '),
                        self.text_buffer.rfind(' - '),
                        self.text_buffer.rfind(': '),
                        self.text_buffer.rfind(' ')
                    ]
                    
                    # Find the best break point
                    break_point = max(break_points)
                    
                    fragment = self.text_buffer[:break_point+1]
                    self.text_buffer = self.text_buffer[break_point+1:]
                    self.sentence_queue.put(fragment)
                    self._info(f"Queued fragment: {fragment}")
                    if not self.first_speech_fragment_finalized:
                        self.time_to_first_speech_fragment = time.time() - self.time_llm_gen_started 
                        self._info(f"\n>> Time to first speech fragment (forced): {self.time_to_first_speech_fragment:.2f} seconds")

                    self.first_speech_fragment_finalized = True

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
                    sentence = self.sentence_queue.get(timeout=0.5)
                    
                    # Wait until not speaking to avoid overlap
                    while self.is_speaking and not self.stop_event.is_set():
                        time.sleep(0.05)
                    
                    if self.stop_event.is_set():
                        break
                    
                    # Speak new sentence
                    self._speak_sentence(sentence, speed=self.speaking_rate)
                    self.sentence_queue.task_done()
                
                except queue.Empty:
                    if self.sentence_queue.empty() and not self.text_buffer and not self.is_speaking:
                        break
        
        finally:
            self.is_processing = False
            
            # If there are still sentences and we're not stopped, restart processor
            if not self.sentence_queue.empty() and not self.stop_event.is_set():
                self._start_sentence_processor()
    
    def _speak_sentence(self, text, speed=1.0, noise_scale=0.667, noise_w=0.8):
        """Synthesize and play a sentence with TTS model."""
        if not text.strip():
            return

        self.is_speaking = True
        
        self._info(f"Speaking: {text}")
        audio_data, sample_rate = self.tts.synthesize(
            text, target_sr = self.sample_rate, 
            speaking_rate=speed, return_as_int16=True)
        self.audio_stream.write(audio_data)

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

    
        # Give a moment for audio to finish playing to ensure user input and agent aren't interfering
        time.sleep(0.5)
    

    def process_prompt(self, user_prompt):
        """Process a prompt through Ollama and stream to Piper."""

        self.messages.append({'role': 'user', 'content': user_prompt})

        if self.verbose:
            self._info(f">> context length: turns: {len(self.messages) / 2}")
            self._info(f">> context length: characters: {len(json.dumps(self.messages))}")

            pretty_json = json.dumps(self.messages, indent=2)
            self._info(f">> Sending prompt to {self.ollama_model_name}: {pretty_json}")


        # reset
        self.assistant_printer.start()
        self.assistant_printer.show_idle('thinking...')

        self.first_speech_fragment_finalized = False
        self.time_llm_gen_started = time.time()
        self.first_chunk_emitted = False

        # get and stream Ollama response to prompt
        response = ollama.chat(
            model=self.ollama_model_name,
            messages=self.messages,
            stream=True,
            options=self.ollama_options
        )
        text_chunks = []
        for chunk in response:
            if not self.first_chunk_emitted:
                self.first_chunk_emitted = True
                self.time_to_first_token = time.time() - self.time_llm_gen_started
                self._info(f"\n>> Time to first token: {self.time_to_first_token:.2f} seconds")

            if self.stop_event.is_set():
                break
                
            if chunk and 'message' in chunk and 'content' in chunk['message']:
                text_chunk = chunk['message']['content']

                self.assistant_printer.show_idle()

                # remove asterisks and other formatting info from the text
                text_chunk = self._clean_llm_output(text_chunk)

                self._process_text_chunk(text_chunk)            
                text_chunks.append(text_chunk)

        assistant_response = ''.join(text_chunks)
        self.messages.append({'role': 'assistant', 'content': assistant_response})

        # print final assistant message
        self.assistant_printer.print(assistant_response, partial=False)

        # Process any remaining text
        self._finish_processing()
            

class AudioToText:
    """Stream from audio and transcribe."""

    def __init__(self, asr_model_name,
                 end_of_utterance_duration=0.5, 
                 min_partial_duration=0.2, max_segment_duration=15,
                 verbose=False,
                printer=None):

        self.verbose = verbose
        self.end_of_utterance_duration = end_of_utterance_duration
        
        self.min_partial_duration = min_partial_duration
        self.max_segment_duration = max_segment_duration
        
        if not printer:
            self.caption_printer = ColoredPrinter("User Input", "blue")
        else:
            self.caption_printer = printer

        # For audio input
        self.audio = None

        self.input_device = captioning_utils.find_default_input_device()
        self._info(f"Using default audio input device: {self.input_device}")
        self.device_index = self.input_device['index']

        # Audio buffer
        self.audio_queue = queue.Queue(maxsize=1000)
        self.input_audio_stream = None

        # Load models
        self.vad = captioning_utils.get_vad(eos_min_silence=200)    
        self.asr_model = captioning_utils.load_asr_model(asr_model_name, 16000, False)

        # Transcription thread
        self.stop_event = threading.Event()
        self.transcription_handler = captioning_utils.TranscriptionWorker(
            sampling_rate=captioning_utils.SAMPLING_RATE)

        
    def start(self):

        # Initialize audio stream
        self.audio = pyaudio.PyAudio()
        self.input_audio_stream = captioning_utils.get_audio_stream(
            self.audio, 
            input_device_index=self.device_index
        )

        # Set stop flag and transcriber thread
        self.stop_event.clear()
        self.transcriber = threading.Thread(target=self.transcription_handler.transcription_worker, 
                                    kwargs={'vad': self.vad,
                                            'asr': self.asr_model,
                                            'audio_queue': self.audio_queue,
                                            'caption_printer': self.caption_printer,
                                            'stop_threads': self.stop_event,
                                            'min_partial_duration': self.min_partial_duration,
                                            'max_segment_duration': self.max_segment_duration})
        self.transcriber.daemon = True
        self.transcriber.start()


    def stop(self):

        # Stop audio stream
        if self.input_audio_stream:
            if self.input_audio_stream.is_active():
                self.input_audio_stream.stop_stream()
            self.input_audio_stream.close()
            self.audio.terminate()
            self.audio = None
        
        # Set stop flag
        self.stop_event.set()

        # Clear audio buffer
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass

        # Stop transcriber thread
        if self.transcriber and self.transcriber.is_alive():
            self.transcriber.join(timeout=3.0)   
            if self.transcriber.is_alive():
                raise Exception("Transcriber thread not stopped within timeout.")
        self.transcription_handler.reset()

        # Reset VAD states
        self.vad.reset_states()


    def mute(self):
        """Temporarily stop the audio input stream without destroying it"""
        if self.input_audio_stream and self.input_audio_stream.is_active():
            self.input_audio_stream.stop_stream()
            self._info("Audio input stream muted")
            return True
        return False
            
    def unmute(self):
        """Resume the audio input stream if it exists"""
        if self.input_audio_stream and not self.input_audio_stream.is_active():
            self.input_audio_stream.start_stream()
            self._info("Audio input stream unmuted")
            return True
        return False
    
    def is_muted(self):
        """Check if the audio input stream is currently muted"""
        if self.input_audio_stream:
            return not self.input_audio_stream.is_active()
        return True  # If no stream exists, consider it muted
            
    def shutdown(self):
        # Clean up audio resources
        if self.input_audio_stream:
            self.input_audio_stream.stop_stream()
            self.input_audio_stream.close()
            self.input_audio_stream = None


    def _info(self, text):
        if self.verbose:
            print(text)

    def get_speech_input(self):
        self.caption_printer.start()

        try:
            self.input_audio_stream.start_stream()
            self._info(f">>> START input audio stream active: {self.input_audio_stream.is_active()}")

            while True:
                # Check for stop signal
                if self.stop_event.is_set():
                    return ""
                
                try:
                    data = self.input_audio_stream.read(captioning_utils.AUDIO_FRAMES_TO_CAPTURE)
                    self.audio_queue.put(data, timeout=0.1)
                except Exception as e:
                    # If we're stopping, just ignore errors
                    if self.stop_event.is_set():
                        return ""
                    # Otherwise, re-raise
                    raise

                all_transcribed = ''
                if not self.transcription_handler.is_speech_recording:
                    if not self.transcription_handler.had_speech:
                        # wait until user spoke - make this message more visible
                        self.caption_printer.print("Please speak now...", partial=True)
                    else:
                        # define EOU when we haven't seen speech for a while
                        if self.transcription_handler.time_since_last_speech() > self.end_of_utterance_duration:
                            self._info(">>> seems user stopped speaking...")
                            # retranscribe and capture all
                            all_transcribed = ' '.join(self.transcription_handler.transcribed_segments)
                            self._info(f">> all said: {all_transcribed}")
                            # reset the transcriber
                            self.transcription_handler.reset()
                            break
        finally:
            # Always try to stop the stream, ignoring errors
            if self.input_audio_stream.is_active():
                self.input_audio_stream.stop_stream()

        return all_transcribed


class VoiceAgent():

    GOODBYE_MESSAGE = 'Good bye!'
    EXIT_COMMAND = 'please exit'

    def __init__(self):
        pass

    def init_AudioToText(self, **audioToTextKwargs):
        self.input_handler = AudioToText(**audioToTextKwargs)

    def init_LLmToAudioOutput(self, **llmToAudiOutputArgsKwargs):
        self.output_handler = LLmToAudio(**llmToAudiOutputArgsKwargs)
    
    def start(self):
        self.output_handler.start()
        self.input_handler.start()

    def stop(self):
        self.output_handler.stop()
        self.input_handler.stop()

    def shutdown(self):
        self.output_handler.shutdown()
        self.input_handler.shutdown()

    def run(self):
        while True:
            if self.stop_event_set():
                break
            
            user_input_transcribed = self.input_handler.get_speech_input()

            if not user_input_transcribed:
                continue

            if self.EXIT_COMMAND.lower() in user_input_transcribed.lower():
                self.output_handler._speak_sentence(self.GOODBYE_MESSAGE)
                time.sleep(0.5)
                break
            else:
                if self.stop_event_set():
                    break

                self.output_handler.process_prompt(user_input_transcribed)     

    def trigger_stop_events(self):
        self.input_handler.stop_event.set()
        self.output_handler.stop_event.set()            
        time.sleep(0.2)

    def stop_event_set(self):
        return self.input_handler.stop_event.is_set() or self.output_handler.stop_event.is_set()
        
    def mute_microphone(self):
        """Temporarily mute the input audio stream"""
        return self.input_handler.mute()
        
    def unmute_microphone(self):
        """Resume the input audio stream"""
        return self.input_handler.unmute()
        
    def is_microphone_muted(self):
        """Check if the microphone is currently muted"""
        return self.input_handler.is_muted()