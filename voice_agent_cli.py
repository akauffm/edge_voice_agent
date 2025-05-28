# CLI for voice agent
# Default settings meant to be run on tiny screen (Raspberry Pi)

import argparse
from voice_agent import VoiceAgent, LLmToAudio

def main():
    """Main function to run the Ollama to Piper streamer."""
    parser = argparse.ArgumentParser(description="Stream text from Ollama to Piper TTS")
    parser.add_argument("--ollama_model_name", default="gemma3:1b", help="Ollama model to use (default: llama3)")
    parser.add_argument("--tts_engine", choices=['piper', 'kokoro'], default="piper", help="which tts engine to use; piper is much faster than kokoro.")
    parser.add_argument("--asr_model_name", default="moonshine_onnx_tiny", help="which asr model to run.")
    parser.add_argument("--tts_model_path", required=False, help="Path to the tts model (.onnx file)")
    parser.add_argument("--speaking_rate", type=float, default=1.0, help="how fast should generated speech be, 1.0 is default, higher numbers mean faster speech")
    parser.add_argument("--max_words_to_speak_start", type=int, default=5, help="maximum number of words to speech onset after a prompt; reduce if latency too high.")
    parser.add_argument("--max_words_to_speak", type=float, default=20, help="always produce speech after this many words were produced ignoring sentence boundaries.")        
    parser.add_argument("--system_prompt", default=LLmToAudio.DEFAULT_SYSTEM_PROMPT, help="Instructions for the model.")
    parser.add_argument("--end_of_utterance_duration", type=float, default=0.5, help="Silence seconds until end of turn of user identified")
    parser.add_argument("--verbose", action="store_true", help="Verbose status info")
    
    args = parser.parse_args()
    
    va = VoiceAgent()

    va.init_LLmToAudioOutput(
        ollama_model_name=args.ollama_model_name,
        system_prompt=args.system_prompt,
        tts_engine=args.tts_engine,
        speaking_rate=args.speaking_rate,
        tts_model_path=args.tts_model_path,
        max_words_to_speak_start=args.max_words_to_speak_start,
        max_words_to_speak=args.max_words_to_speak,
        verbose=args.verbose
    )

    va.init_AudioToText(
        asr_model_name=args.asr_model_name,
        end_of_utterance_duration=0.5, 
        verbose=args.verbose
    )
    va.start()

    va.run()

if __name__ == "__main__":
    main()
