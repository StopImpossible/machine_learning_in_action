from math import log
import operator

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
    print base_entropy
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

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.key():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(),
                                key = operator.itemgetter(1),
                                reverse = True)
    return sorted_class_count

def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label:{}}
    del(labels[best_feature])
    feat_values = [example[best_feature] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(spit_data_set(data_set, best_feature, value), sub_labels)
    return my_tree

my_data, labels = create_data_set()
my_tree = create_tree(my_data, labels)
print my_tree