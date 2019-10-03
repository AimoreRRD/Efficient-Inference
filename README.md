# NamePredict
 This module predicts what class an arbitrary name belongs: 
```
1:'Company',
2:'EducationalInstitution',
3:'Artist',
4:'Athlete',
5:'OfficeHolder',
6:'MeanOfTransportation',
7:'Building',
8:'NaturalPlace',
9:'Village',
10:'Animal',
11:'Plant',
12:'Album',
13:'Film',
14:'WrittenWork'
```

### Install Dependencies
- conda env create -f NP.yml --name NP
- pandas should be 0.24.1 and scipy 1.1.0
- conda install -c conda-forge pandas_ml
- conda install -c conda-forge matplotlib
- conda install -c anaconda seaborn


### Train
- Use NamePredict.ipynb

### Run predict a name (examples)
- python run_predict_name.py run --name Jhonson Brothers 
- python run_predict_name.py run --name Coca Cola
- python run_predict_name.py run --name Hello
- python run_predict_name.py run --name Streetbee
