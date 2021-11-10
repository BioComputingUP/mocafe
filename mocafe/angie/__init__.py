import random
import pickle


# init random state
def set_random_state(randomstate):
    random.setstate(randomstate)


# get random state
def get_random_state():
    return random.getstate()


# load random state from file
def load_random_state(file_name):
    with open(file_name, "rb") as f:
        random.setstate(pickle.load(f))


# save random state to file
def save_random_state(file_name):
    with open(file_name, "wb") as f:
        pickle.dump(random.getstate(), f)
