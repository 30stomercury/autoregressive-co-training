# Autoregressive Co-training

The implementation of the paper:

[Autoregressive Co-Training for Learning Discrete Speech Representations](https://arxiv.org/abs/2203.15840) \
Sung-Lin Yeh, [Hao Tang](https://homepages.inf.ed.ac.uk/htang2/index.html)

# Dependencies
```
pip install -r requirements.txt
```

# Models
The co-training model described in the paper is defined in `cotraining.py`. Different components of the model
are modular and can be easily modified.

# Data
Data are processed to [Kaldi I/O form](https://kaldi-asr.org/doc/io.html), 
which uses scp files to map utterance ids to positions in ark files. Functions used to process `.scp` and `.ark` files 
can be found under `dataflow/`. We provide a data sample in `sample/` for user to run the pipeline. User can simply pluge in 
your custom dataloader [here](https://github.com/30stomercury/autoregressive-co-training/blob/d88d7e6c3f02085be8f0698ba2fde667b4d5349e/train.py#L36).

# Train
```
python3 train.py --config config/cotraining.yaml
```
# Pre-trained Models
Hours  | Num codes | Model | dev93 (PER) | eval92 (PER) | Link
:-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
360  | 256 | Marginalization | 19.5 | 19.0 | [link](https://drive.google.com/drive/u/1/folders/18lNV9Fz1ypnTo7Ivx0nVTFeV1EGk-avd)
960  | 256 | Marginalization | 18.2 | 17.8 | [link](https://drive.google.com/drive/u/1/folders/12KuEn9UBTUBl2qetZLypXsHPPEQGYmby) 
