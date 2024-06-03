# THREAD: Thinking Deeper with Recursive Spawning

## Setup
The code to test THREAD on each benchmark is provided in the corresponding folder. For OpenAI models, please set the variable `API_KEY` to your OpenAI API key. For models from HuggingFace, please set the variable `HF_auth_token` to your HuggingFace User Access Token. 

## Dependencies and required datasets
For ALFWorld and WebShop, install and set up the environments from the source repositories [alfworld](https://github.com/alfworld/alfworld), [webshop](https://github.com/princeton-nlp/WebShop). For TextCraft, please download the [crafting recipes](https://github.com/InventivetalentDev/minecraft-assets/tree/1.16.5/data/minecraft/recipes) and store them in `textcraft/textcraft/recipes`. For the DataCommons QA benchmark, the dataset is provided directly. Due to restrictions with MIMIC-III data access, we cannot directly release the dataset for MIMIC-III ICU QA. However, upon gaining access to MIMIC- III following the instructions outlined at [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/), you can reproduce the full dataset using the code the code provided here. Once you have access, you can build the in-hospital-mortality dataset following the steps outlined here: [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). Following this, you should have a folder named `in-hospital-mortality`. Please set the MIMIC_DATA_DIR variable in `mimiciii_icu_q/build_mimiciii_icu_qa.py` to the path of this folder. `build_mimiciii_icu_qa.py` then creates the full MIMIC-III ICU QA dataset.



