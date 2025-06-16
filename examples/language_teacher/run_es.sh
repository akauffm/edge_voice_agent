python voice_agent_ui.py \
   --ollama_model_name gemma3:1b \
   --tts_model_path es_ES-carlfm-x_low.onnx  \
   --asr_model_name whisper_tiny \
   --language es \
   --system_prompt "`cat examples/language_teacher/instructions_es.txt `" \
   --start_message "Buenas dias! Soy tu maestra de español y estoy súper emocionada de aprender contigo hoy. ¿Qué quieres practicar?" \
   --min_partial_duration 0.5 \
   --end_of_utterance_duration 0.75 \
   --speaking_rate 0.75
