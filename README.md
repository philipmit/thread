# THREAD: Thinking Deeper with Recursive Spawning

## Dependencies and required datasets
For ALFWorld and WebShop, install and set up the environments from the source repositories [alfworld](https://github.com/alfworld/alfworld), [webshop](https://github.com/princeton-nlp/WebShop). For TextCraft, please download the [crafting recipes](https://github.com/InventivetalentDev/minecraft-assets/tree/1.16.5/data/minecraft/recipes) and store them in `textcraft/textcraft/recipes`. For the MIMIC-III ICU QA benchmark, you first need to build the in-hospital-mortality dataset following the steps outlined here: [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). Once this is complete, you should have a folder named `in-hospital-mortality`. Please set the MIMIC_DATA_DIR variable in `mimiciii_icu_q/run_mimiciii_icu_qa.py` to the path of this folder.



