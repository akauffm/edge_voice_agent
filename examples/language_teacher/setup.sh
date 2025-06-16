# (1) Download LLama model
ollama pull llama3.2:1b

# (1) Download piper voices

# best voices from https://huggingface.co/rhasspy/piper-voices for other languages
# many of the other voices here produce very poor audio, hallucinate or mumble considerably.

# DE
wget https://huggingface.co/rhasspy/piper-voices/blob/main/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx
wget https://huggingface.co/rhasspy/piper-voices/blob/main/de/de_DE/thorsten/low/de_DE-thorsten-low.onnx.json

# ES
wget https://huggingface.co/rhasspy/piper-voices/blob/main/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx
wget https://huggingface.co/rhasspy/piper-voices/blob/main/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json


