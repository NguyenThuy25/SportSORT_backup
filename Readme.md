# SportSORT

This is the official code for paper "SportsSORT: Overcoming Challenges of Multi-Object Tracking in Sports through Domain-Specific Features and Out of view Re-Association"

## Setup Instructions
- Clone this repo
- Install dependencies.
```
conda create -n sport_sort python=3.7
conda activate sport_sort
cd SportSORT/reid
pip install -r requirements.txt
pip install cython_bbox
python setup.py develop
cd ..
```


## Inference on SportsMOT dataset
Run on train set
```
cd SportSORT/SportSORT
python tools/infer_test2.py --split train --online
```

## Implementation Detail
Our main method is implemented in file "/SportSORT/SportSORT/tracker/SportSORT.py"
