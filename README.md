CZ4042 Project

Dataset preparation: 
    Download and unzip the aligned image Adience dataset from https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender
    Run 'python3 img_name_clean.py' to clean the image file names.

Repo stucture:
    - aligned
        - ...
    - train_test_split
        - X_train.csv
        - ...
    - img_name_clean.py
    - model.py
    - main.py 
    - Age_embedding.ipynb
    - Data_preprocessing.ipynb
    - logs
        -...
    - README.md

./aligned/
    the dataset folder

./train_test_split/
    includes X_train.csv X_test.csv X_train_age.csv X_test_age.csv y_train.csv y_test.csv
    the suffix '_age' also includes the age_embedding features

./*.ipynb
    the dataset preprocessing notebook, which created the csv files in ./train_test_split

./logs/
    the training stdout logs for all our experiments

To run our code:
    e.g. 'python3 main.py ./aligned/ -a=resnet18 --crop=random --age=True --optim=adamw'
    if set age==True, the model will utilise the age information, else not
    more command arguments please refer to our code in main.py