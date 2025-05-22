# Exemplary Voice Agent running fully on-device

* flexible wrt to ASR, LLM and TTS component, currently supported:
   * ASR: Moonshine, FasterWhisper, Nemo FastConformer, Vosk
   * TTS: Piper, Kokoro
   * LLM: anything through Ollama
* components chosen to work fully offline on-device, CPU only
   * default setup can run on Raspberry Pi 5
      * ASR: Moonshine tiny
      * TTSL: Piper
      * LLM: Gemma3:1b

* simplified end of utterance detection based on Silero VAD (set the duration via ```---end_of_utterance_duration```, > 0.5 seems to be good)

* currently no interruption capabilities

## Installation

* then install requirements here
```pip install -r requirements.txt```

* download these two repos and install with ```pip install -e .``` each (also install their dependencies)
   * https://github.com/ktomanek/captioning (install all ASR models you want to run, see instructions there)
   * https://github.com/ktomanek/edge_tts_comparison

* download sentence splitter: ```python -c "import nltk; nltk.download('punkt_tab')```

### ASR models

* moonshine:
```pip install useful-moonshine-onnx@git+https://git@github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx```

* optionally install nemo asr models: ```pip install "nemo_toolkit[asr]"```

* optionally install faster whisper: ```pip install faster whisper```

### TTS models

* if using Kokoro, copy model files here
* for Piper, Lessac voice included here, copy other voice models if wanted

### Ollama

* install ollama locally: https://ollama.com/download
* then pull the model you want ot use, eg: 

```ollama pull gemma3:1b```

* then install [ollama python library](https://github.com/ollama/ollama-python) 

```pip install ollama```


## Examples


### Survival Guide

```python voice_agent.py \
  --system_prompt "You are an outdoor survival guide assistant helping users, who have no internet access, no phone access, and are far from civilization to deal with challenges they experience in the outdoors. Please give helpful advice, but be VERY brief. Only provide details when explicitly asked for it." \
  --asr_model_name moonshine_onnx_base \
  --speaking_rate 2.0 \
  --end_of_utterance_duration 0.7
  ```

## Configure system prompt from text

* you can increase the speaking rate to make long responses not feel quite as length

```python voice_agent.py --speaking_rate 3.0 --system_prompt "`cat cat_specialist.txt`" ```

## Other models

* moonshine base seems to run fast enough on Raspberry Pi.
* Gemma3:4b leads to significant improvement on conversation side, but starts feeling slow

```python voice_agent.py --asr_model_name moonshine_onnx_base --ollama-model-name gemma3:4b --speaking_rate 3.0```
