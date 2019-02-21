# COMP 7950 (Deep Learning)

## Setup
1. Create a python environment using either `virtualenv` or `anaconda`.
2. Install the dependencies `pip install -r requirements.txt` after activating the environment.
3. Change the hyperparameters in `config/configuration.json` to suit the training.
4. Initiate the training by `python train_localize.py`
5. Test the trained model by `python test_localize.py -pre ./path_of_the_model`
6. To visualize the results, run `python visualize_results.py` by changing relevant variables. 
