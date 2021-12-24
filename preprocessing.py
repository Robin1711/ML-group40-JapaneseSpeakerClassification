import data_loader
import argparse

def kmeans():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load train/test data in a specific format')
    #parser.add_argument('-labels_per_timestep', type=bool, default=False, help='if added each time step of the utterance has a label, otherwise the label is associated with the entire timeserise of the utterance.')

    args = parser.parse_args()

    