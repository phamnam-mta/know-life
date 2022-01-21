# Vietnamese healthcare Knowledge Base System

The system would be as following
![image](asset/KBS.png)


## Setup

1. Clone BioSyn repo
This repo support Entity Linking (EL) for KB systems
```
git clone https://github.com/dmis-lab/BioSyn.git
```
2. Installation

```
pip install -r requirements.txt
```
3. Download checkpoint

- Download NER Checkpoint in [here](https://drive.google.com/drive/folders/111ThBNm1B744V5WnsQDU64ou5oAyK9j_?usp=sharing)

- Copy downloaded checkpoint into `./ckpt` folder

## Usage
### 1.1 Informative QA

```
python example.py    
```