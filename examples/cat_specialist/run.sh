python ../../voice_agent_cli.py \
   --ollama_model_name gemma3:1b \
   --tts_model_path ../../en_US-lessac-low.onnx  \
   --system_prompt "`cat instructions.txt`" \
   --start_message "I'm here to listen and understand whatever you and your feline friend are going through right now." \
   --speaking_rate 1.5
