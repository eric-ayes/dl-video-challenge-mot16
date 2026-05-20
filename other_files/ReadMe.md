# Other files that were used to achieve results and do testing

This directory contains all other files we created to create other models but did not lead to the final best approach.\
It is basically an adapted version of the baseline code with other experiments.\
Keep in mind that the file structure of `dlvsp_challenge_material` needs exist to put the content of this directory into `src/`.\
Filepaths need to be changed according to the user's systems.\
A `requirements.txt` with all dependencies is added as well. It is based on a Python 3.10.19 conda environment executed on Mac with ARM architecture.

Apart from the baseline files, there are other files based on the code in `tracker/`:
- `finetuning_crowdhuman.ipynb`: Finetuning file for the FRCNN-FPN model, was created and executed in Kaggle.
- `make_video.py`: A file to create an `.mp4` video based on video and prediction / ground truth reference.
- `model_exec.ipynb`: A version of the baseline notebook that was used locally.
- `mot_eval.py`: A python file based on the baseline notebook used for evaluation (created to not having to execute everything manually).
