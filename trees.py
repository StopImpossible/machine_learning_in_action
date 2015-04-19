from math import log

def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shanno_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shanno_ent -= prob * log(prob, 2)
    return shanno_ent

def create_data_set():
    data_set = [
        [1, 1, 'Yes'],
        [1, 1, 'Yes'],
        [1, 0, 'No'],
        [0, 1, 'No'],
        [0, 1, 'No']
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

def spit_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set

def choose_best_feature_to_split(data_set):
    num_feature = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feature):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = spit_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if(info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

my_dat, labels = create_data_set()
shanno = choose_best_feature_to_split(my_dat)
print shanno