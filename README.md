# MeRP: Efficient Multi-Expert Routing Prompt Encoding for Multilingual Translation with LLMs
This repository provides the implementation of **MeRP** for multilingual machine translation with LLMs.
## 1. Dataset Preparation
Please place the processed datasets in:
```
data/translation_data/
```
### Training Set
```
{
  "instruction": "Like honestly, head to bed very soon. I hope you have a very good rest of your night.",
  "input": "",
  "output": "Только честно, идите спать очень скоро. Я надеюсь, у вас будет очень хороший остаток ночи.",
  "lang_pair": "en-ru"
}
```
### Test Set
```
{  "src": "Nyní máme čtyřměsíční myši bez cukrovky, které ji dříve měly,“ dodal.",  "ref": "We now have 4-month-old mice that are non-diabetic that used to be diabetic, he added.",  "lang_pair": "cs-en"}
```
## 2. Environment
This project is built on the **LLaMA-Factory** framework.
```
conda create -n MeRP python=3.10.20
conda activate MeRP

cd MeRP
pip install -e .
pip install -r requirements.txt

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```
## 3. Configuration
Training configuration:
```
examples/train_lora/gemma2_2b_prompt_multi.yaml
```
Inference configuration:
```
eval_mulit/configs/merp_gemma2.yaml
```
## 4. Run
Run the complete training, inference, and evaluation pipeline:
```
chmod +x examples/train_lora/run.sh && ./examples/train_lora/run.sh
```
