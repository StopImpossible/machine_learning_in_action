from numpy import *
import operator
from os import listdir

def creat_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis = 1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_ilabel = labels[sorted_dist_indicies[i]]
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def file2_matrix(filename):
    fr = open(filename)
    number_of_lines = len(fr.readlines())
    print number_of_lines
    return_mat1 = zeros((number_of_lines, 3))
    class_label_vector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat1[index, :] = list_from_line[0:3]
        class_label_vector.append(list_from_line[-1])
        index += 1
    return return_mat1, class_label_vector

'''data_normalizing code'''
def auto_norm(data_set):
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    ranges = max_value - min_value
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_value, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_value

def dating_data_test():
    hoRatio = 0.10
    dating_data_mat, dating_labels = file2_matrix("dating_test_set.txt")
    norm_mat, ranges, min_value = auto_norm(dating_data_mat)
    print min_value
    m = norm_mat.shape[0]
    num_test_vecs = int(m * hoRatio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs: m, :], dating_labels[num_test_vecs: m], 3)
        if (classifier_result != dating_labels[i]): error_count += 1
    print error_count / float(num_test_vecs)

def img2vector(filename):
    return_vect = ((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect

def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector(r'trainingDigits\\' + file_name_str)
    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(test_file_list):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector(r'testDigits\\' + file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num_str)
        if (classifier_result != class_num_str):
            error_count += 1
    print "\n the total number of errors is: %d" % error_count
    print "\n the total error rate is: %f" % (error_count / float(m_test))

handwriting_class_test()