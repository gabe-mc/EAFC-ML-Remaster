
<img width="1367" alt="eafc-banner" src="https://github.com/user-attachments/assets/0a921152-ce5e-4afe-be1c-627fb96b23a5" />

[![EA](https://img.shields.io/badge/EA-%23000000.svg?logo=ea&logoColor=white)](https://www.ea.com/sports)
[![Python](https://img.shields.io/badge/Python-blue?logo=python&logoColor=fff)](https://www.python.org/)
![Repo size](https://img.shields.io/github/repo-size/gabe-mc/3d-pose-detection?color=green)
[![GitHub last commit](https://img.shields.io/github/last-commit/gabe-mc/EAFC-ML-Remaster?color=lightgrey)](https://github.com/gabe-mc/EAFC-ML-Remaster/commits)
[![License](https://img.shields.io/github/license/gabe-mc/EAFC-ML-Remaster?color=blue)](https://github.com/gabe-mc/EAFC-ML-Remaster/blob/main/LICENSE)

# EAFC-ML-Remaster

Our overarching goal is to make a mod to EAFC that will be entirely separate from the game itself, only operating on data that is able to be collected from a video of the game. This will include a collection of different “mods”, including, but not limited to:

- Upgraded commentary, with famous announcers/celebrities doing the voice  
- Updated graphics, mirroring actual graphics found on SkySports or FuboTV  
- ... any other ideas that can be accomplished without actually plugging into the game ...

With more time, we can also look into models to actually make the graphics way better.



## Upgraded Commentary

### Commentators
- Conor McNamara and Jim Beglin  
- Best two Premier League announcers, most prominent right now, will give the most realism.

### Workflow

To accomplish this, we will need:

- A bounding box computer vision model that can track individual players  
  - Should keep tabs on players even after they go off screen and return.  
  - One approach: fine-tune model on specific players, always play with the same teams.  
  - If players are visually distinct, use bounding-box detection + image embedding (e.g., ViT or CLIP):  
    - https://huggingface.co/models?pipeline_tag=image-feature-extraction&sort=trending  
  - Could also fine-tune a YOLO-style model for detection:  
    - https://docs.ultralytics.com/tasks/classify/  
  - Replays are a challenge due to no visible players on screen.

- A bounding box model that can tightly fit the ball with high update rate  
  - Ideally same model as for players, to avoid multiple passes.

- A LLM fine-tuned on commentary data from Conor McNamara and Jim Beglin  
  - May require pre-labeling via a larger LLM (e.g., `{“And that’s a great shot!” : ball is shot}`)  
  - Prompting strategy:  
    - “Respond as though you are an announcer commentating on the following action in a football/soccer game: <x>”

- A way to differentiate the two commentators' voices  
  - Fine-tune audio models with samples of their voices:

    - Fish Speech (high quality, heavy):  
      - https://huggingface.co/fishaudio/fish-speech-1.5  
    - Zonos (decent quality, lighter):  
      - https://github.com/Zyphra/Zonos  
    - Kokoro-82M (very lightweight, not sure about fine-tuning):  
      - https://huggingface.co/hexgrad/Kokoro-82M

## LLM Fine-Tuning

### Data

- Gather data from Premier League games  
- (Small dataset sourced from sketchy sites)

### Workflow

1. Separate each commentator's voice into sub-clips  
2. Transcribe each voice separately using Whisper  
3. Train a voice model on each  
4. Fine-tune an LLM to speak in their style  

Alternative approach: few-shot prompting using a small model like Phi 3.5


## Computer Vision Model

- Ultralytics provides good resources and regular YOLO updates:  
  - https://docs.ultralytics.com/models/

> Stay away from YOLO-World and YOLO variants that integrate NMS into the model itself.


