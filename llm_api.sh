#!/bin/bash
# source envllm/bin/activate
#pip install time requests pandas tqdm
python3 llm_api$1.py guidelines/guidelines-no-examples.txt examples $1 > log.txt
python3 llm_api$1.py guidelines/guidelines-no-examples.txt >> log.txt
python3 llm_api$1.py guidelines/guidelines-dummy.txt examples $1 >> log.txt
python3 llm_api$1.py guidelines/guidelines.txt examples>> log.txt
python3 llm_api$1.py guidelines/guidelines.txt examples $1 >> log.txt
python3 llm_api$1.py guidelines/guidelines-dummy.txt >> log.txt
