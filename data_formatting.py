"""
Data analysis, but more preprocessing eigenlijk
includes methods for
- Truncation of the data
- Padding data (with 0's or noise)
- Transposing data-matrix from no_energies x channels => channel x no_energies
- Transforming labelvectors to numeric labels: [[1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0]] => [0,2]
"""

from itertools import chain
import preprocessing

CHANNELS = 12
LABEL_TYPE = "timeseries"
PAD_LENGTH_TRAIN = 26           # Pad at most to this number. Deducted from max spoken vowel length in training set
TRUNCATE_LENGTH_TRAIN = 26      # Truncate at least to this number. Deducted from max spoken vowel length in training set
SIGNAL_LENGTH = 26

# NOTE
# In the testing set, there are examples (vowels) that have a length that is greater than 26.
# Do we assume we have the maximal length spoken vowel in the training set? No. (longest test set vowel == 29)
# We do think we might encounter longer vowels. However..
# Our assumption: we cannot train a NN on training examples, where all last values are padded values (i.e. have no significant value)


"""
Pad the timeseries channel energies
Args:
    data ( [[[ double ]]] ):    List of spoken vowels. Each vowel is represented by an unknown length list, each element
                                being a list of length 12 containing the energies of the channels on that "i-th" index.
                                Note this is the non-transposed data returned by
                                preprocessing.get_data(label_type="timeseries")
    padding_type ( string ):    String ("BACK" | "FRONT") specifying where the padding should happen
                                ([2,2] =>  [2,2,0,0] OR [0,0,2,2])
                                Defaults to "BACK" 
    value ( string ):           String ("ZEROES" | "NOISE") Specifying what to pad the data with. Noise will create values
                                that are close to 0.0, but have small deviations.
                                Defaults to "ZEROES"
                                => Not implemented yet.
Returns:
    padded_data ([[[ double ]]]): Input, where each entry is padded, such that the length are all equal to the maximum length
"""
def pad_signals(data, padding_type="BACK", value="ZEROES", padding_to=PAD_LENGTH_TRAIN):
    padded_data = list()
    for entry in data:
        needed_padding = padding_to - len(entry)
        if value == "ZEROES":
            pad_entry = [0.0 for _ in range(0, CHANNELS)]
            padded_entry = entry + needed_padding * [pad_entry] if padding_type == "BACK" else needed_padding * [pad_entry] + entry
        # else: # value == "NOISE"
        #     padded_entry = entry
        #     for x in needed_padding
        padded_data.append(padded_entry)

    return padded_data


"""
Truncate the timeseries channel energies

Args:
    data ( [[[ double ]]] ):    List of spoken vowels. Each vowel is represented by an unknown length list, each element
                                being a list of length 12 containing the energies of the channels on that "i-th" index.
                                Note this is the non-transposed data returned by
                                preprocessing.get_data(label_type="timeseries")
    truncate_type ( string ):    String ("BACK" | "FRONT") specifying where the truncating should happen
                                ([2,2,0,0] =>  [2,2] OR [0,0])
                                Defaults to "BACK"
Returns:
    padded_data ([[[ double ]]]): Input, where each entry is padded, such that the length are all equal to the maximum length
"""
def truncate_signals(data, truncate_to=TRUNCATE_LENGTH_TRAIN, truncation_type="BACK"):
    truncated_data = list()
    for entry in data:
        needed_truncation = len(entry) - truncate_to
        if needed_truncation > 0:
            truncated_entry = entry[0:truncate_to] if truncation_type == "BACK" else entry[needed_truncation:]
            truncated_data.append(truncated_entry)
        else:
            # No truncation needed, but we still need to incorporate the entry in the final result
            truncated_data.append(entry)

    return truncated_data

"""
Transposes the timeseries matrix
N x no_energies x channel 

Args:
    data ( [[[ double ]]] ): list of spoken vowels. Each vowel is represented by an unknown length list, each element
                            being a list of length 12 containing the energies of the channels on that "i-th" index

Returns:
    transposed_data ([[[ double ]]]): list of spoken vowels. Each vowel is represented by 12 lists. Each list containing all
                            energies of that channel.  
"""
def transpose(data):
    transposed_data = list()
    for entry in data:
        transposed_entry = [list() for _ in range(0, CHANNELS)]
        for energies in entry:
            for channel, energy in enumerate(energies):
                transposed_entry[channel].append(energy)
        transposed_data.append(transposed_entry)

    return transposed_data

"""
Transforms labelvector to a list of numeric labels
    [[1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0]] => [0,2]

Args:
    labelvectors ([[double]]): list of one-hot encoded labelvectors

Returns:
    transformed data ([int]): list of numeric labels
"""
def transform_labelvectors_to_labels(labelvectors):
    y_label = list()
    for labelvector in labelvectors:
        label = [index for index, label in enumerate(labelvector) if label != 0.0]
        y_label.append(label)
    return list(chain(*y_label))


def pad_truncate_transpose_data(data, signal_length=SIGNAL_LENGTH):
    data_padded = pad_signals(data, "BACK", padding_to=signal_length)
    data_padded_truncated = truncate_signals(data_padded, truncate_to=signal_length)
    data_padded_truncated_transposed = transpose(data_padded_truncated)
    return data_padded_truncated_transposed

"""
Data formatting demo of padding, truncation, and transposing
"""
if __name__ == '__main__':
    D = preprocessing.get_data(label_type=LABEL_TYPE)
    x_train, y_train, x_test, y_test = D

    x_train_padded = pad_signals(x_train, "BACK")
    x_train_padded_truncated = truncate_signals(x_train_padded, truncate_to=18)
    x_train_padded_truncated_transposed = transpose(x_train_padded_truncated)

    print(f"For signal 9:")
    for channel, energies in enumerate(x_train_padded_truncated_transposed[9]):
        print(f"\tCh: {channel}, len: {len(energies)} -> {energies}")
