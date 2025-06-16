Minimal language teacher experiment for ES and DE.

* ASR: whisper tiny
* LLM: gemma3:1b
   * note: we found llama3.2 (1b) to have better quality for de and es conversations, but inference speed is significantly slower (8 token/sec vs 13 token/sec for Gemma3)
* TTS: piper -- seleced voices that sound most realistic (we still see hallucinations and quite robotic output though...)

Run `setup.sh` first to download necessary models.

