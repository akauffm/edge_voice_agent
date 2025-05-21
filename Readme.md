# Install

* download these two repos and install with ```pip install -e .``` each
   * https://github.com/ktomanek/captioning
   * https://github.com/ktomanek/edge_tts_comparison

* then install requirements here
```pip install -r requirements.txt```

## ASR models

```pip install useful-moonshine-onnx@git+https://git@github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx```

* also install piper
* optionally install kokoro ```pip install kokoro```
* optionally install nemo asr models: ```pip install "nemo_toolkit[asr]"```

## Ollama

* install ollama locally: https://ollama.com/download
* then pull the model you want ot use, eg: 

```ollama pull gemma3:1b```

* then install [ollama python library](https://github.com/ollama/ollama-python) 

```pip install ollama```

## download NLTK sentence splitter

* download sentence splitter: ```python -c "import nltk; nltk.download('punkt_tab')```

# Run

## Survival Guide

```python voice_agent.py \
  --system_prompt "You are an outdoor survival guide assistant helping users, who have no internet access, no phone access, and are far from civilization to deal with challenges they experience in the outdoors. Please give helpful advice, but be VERY brief. Only give details when asked." \
  --speaking_rate 2.0 \
  --end_of_utterance_duration 0.7
  ```

## Configure system prompt from text

```python voice_agent.py --speaking_rate 3.0 --system_prompt "`cat cat_specialist.txt`" ```

## Other models

```python voice_agent.py --asr_model_name moonshine_onnx_base --ollama-model-name gemma3:4b --speaking_rate 3.0```
