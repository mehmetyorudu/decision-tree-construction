import math
import csv
from collections import Counter, defaultdict

# calculate entropy
def entropy(data):
    total = len(data)
    if total == 0:
        return 0
    label_counts = Counter([row[-1] for row in data])
    ent = 0
    for count in label_counts.values():
        prob = count / total
        ent -= prob * math.log2(prob)
    return ent

# split the dataset
def split_data(data, column_index):
    splits = defaultdict(list)
    for row in data:
        key = row[column_index]
        splits[key].append(row)
    return splits

# Calculate information gain for considerable column
def information_gain(data, column_index):
    total_entropy = entropy(data)
    splits = split_data(data, column_index)
    weighted_entropy = sum((len(subset)/len(data)) * entropy(subset) for subset in splits.values())
    return total_entropy - weighted_entropy

# Karar ağacını oluştur
def build_tree(data, headers):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):  # if classes are same, leaf
        return labels[0]

    if len(headers) == 1:  
        return Counter(labels).most_common(1)[0][0]

    gains = [information_gain(data, i) for i in range(len(headers)-1)]
    best_index = gains.index(max(gains))
    best_feature = headers[best_index]

    tree = {best_feature: {}}
    splits = split_data(data, best_index)

    for value, subset in splits.items():
        new_headers = headers[:best_index] + headers[best_index+1:]
        new_subset = [row[:best_index] + row[best_index+1:] for row in subset]
        tree[best_feature][value] = build_tree(new_subset, new_headers)

    return tree

# print decision tree
def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for key, branch in tree.items():
            for value, subtree in branch.items():
                print(f"{indent}[{key} = {value}]")
                print_tree(subtree, indent + "  ")
    else:
        print(indent + "-> " + tree)

# take input and predict 
def predict(tree, headers, sample):
    if not isinstance(tree, dict):
        return tree
    for key in tree:
        feature_index = headers.index(key)
        value = sample[feature_index]
        branch = tree[key].get(value)
        if branch is None:
            return "Unknown"
        return predict(branch, headers, sample)

# read csv
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        headers = lines[0]
        data = lines[1:]
    return headers, data


def main():
    filename = input("Enter name of the csv file: ")  # Example: weather.csv
    headers, data = read_csv(filename)
    tree = build_tree(data, headers)
    print("\nDecision Tree:")
    print_tree(tree)

    while True:
        proceed = input("\nWould you like to try another sample (y/n): ").lower()
        if proceed == 'n':
            break
        elif proceed == 'y':
            print("\nEnter new sample: ")
            sample = []
            for feature in headers[:-1]:
                val = input(f"{feature}: ")
                sample.append(val)
            result = predict(tree, headers, sample)
            print(f"Guess: {headers[-1]} = {result}")
        else:
            print("Please enter only 'y' or 'n'.")

if __name__ == "__main__":
    main()
