# (1) Download piper voices

# best voices from https://huggingface.co/rhasspy/piper-voices for other languages
# many of the other voices here produce very poor audio, hallucinate or mumble considerably.

# DE
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/karlsson/low/de_DE-karlsson-low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/karlsson/low/de_DE-karlsson-low.onnx.json

# ES
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json

# (2) Optionally download LLama model
# ollama pull llama3.2:1b

