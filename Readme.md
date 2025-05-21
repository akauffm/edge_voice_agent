# Install

* download these two repos and install with ```pip install -e .``` each
   * https://github.com/ktomanek/captioning
   * https://github.com/ktomanek/edge_tts_comparison

* then install requirements here
```pip install -r requirements.txt```

```pip install useful-moonshine-onnx@git+https://git@github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx```

* also install piper
* optionally install kokoro ```pip install kokoro```
* optionally install nemo asr models: ```pip install "nemo_toolkit[asr]"```

# Run

```python voice_agent.py \
  --system_prompt "You are an outdoor survival guide assistant helping users, who have no internet access, no phone access, and are far from civilization to deal with challenges they experience in the outdoors. Please give helpful advice, but be VERY brief. Only give details when asked." \
  --speaking-rate 3.0
  ```

