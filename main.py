from sklearn import tree
import pandas
import numpy

def accuracy(test, source):
    test_list = list(test)
    source_list = list(source)
    num_of_correct = 0
    for i in range(len(source_list)):
        if source_list[i] == test_list[i]:
            num_of_correct += 1
    return num_of_correct / len(source_list)


def bagg(sample, num):
    global labels, decision_tree_classifier
    target_list = sample["target"]
    features_list = sample[labels]

    decision_tree_classifier = tree.DecisionTreeClassifier()
    decision_tree_classifier.fit(features_list, target_list)
    return decision_tree_classifier.predict(test_features)
    # print("Bagging Accuracy For Sample", num, ":", accuracy(predictions, test_target))


def make_tree_for_forest(sample):
    random_features_indexes = numpy.random.choice(range(len(labels)), 5, replace=False)
    randomly_chosen_labels = labels[random_features_indexes]
    train_set_forest = sample[randomly_chosen_labels]
    test_features_forest = test_features[randomly_chosen_labels]

    decision_tree_classifier = tree.DecisionTreeClassifier()
    decision_tree_classifier.fit(train_set_forest, sample["target"])
    return decision_tree_classifier.predict(test_features_forest)

def vote(predictions_1, predictions_2, predictions_3, predictions_4, predictions_5):
    voter = {0:0, 1:0}
    total_predictions = []
    for i in range(len(predictions_1)):
        voter[predictions_1[i]] += 1
        voter[predictions_2[i]] += 1
        voter[predictions_3[i]] += 1
        voter[predictions_4[i]] += 1
        voter[predictions_5[i]] += 1

        if voter[0] > voter[1]:
            total_predictions.append(0)
        else:
            total_predictions.append(1)

        voter = {0:0, 1:0}
    return total_predictions


dataset = pandas.read_csv("data.csv")
labels = dataset.columns.values
labels = numpy.delete(labels, len(labels)-1)

train_index = numpy.random.rand(len(dataset)) < 0.8

train_set = dataset[train_index]
test_set = dataset[~train_index]

train_target = train_set["target"]
train_features = train_set[labels]

test_target = test_set["target"]
test_features = test_set[labels]


decision_tree_classifier = tree.DecisionTreeClassifier()
decision_tree_classifier.fit(train_features, train_target)

predictions = decision_tree_classifier.predict(test_features)

print("Decision Tree Accuracy:", accuracy(predictions, test_target))

print("-----------------------------")

# bagging
sample_1 = train_set.sample(n=150, replace=True)
sample_2 = train_set.sample(n=150, replace=True)
sample_3 = train_set.sample(n=150, replace=True)
sample_4 = train_set.sample(n=150, replace=True)
sample_5 = train_set.sample(n=150, replace=True)

predictions_1 = bagg(sample_1, 1)
predictions_2 = bagg(sample_2, 2)
predictions_3 = bagg(sample_3, 3)
predictions_4 = bagg(sample_4, 4)
predictions_5 = bagg(sample_5, 5)
predictions = vote(predictions_1, predictions_2, predictions_3, predictions_4, predictions_5)
print("Bagging Accuracy:", accuracy(predictions, test_target))

print("-----------------------------")

# removing features one by one
labels_list = list(labels)
for label in labels_list:
    new_train_features = train_features.drop(columns=[label])
    new_test_features = test_features.drop(columns=[label])

    decision_tree_classifier = tree.DecisionTreeClassifier()
    decision_tree_classifier.fit(new_train_features, train_target)

    predictions = decision_tree_classifier.predict(new_test_features)

    print("Accuracy after removing ", label, ":", accuracy(predictions, test_target))

print("-----------------------------")

# randomly choosing 5 features
random_features_indexes = numpy.random.choice(range(len(labels)), 5, replace=False)
randomly_chosen_labels = labels[random_features_indexes]

train_features_random = train_features[randomly_chosen_labels]
test_features_random = test_features[randomly_chosen_labels]

decision_tree_classifier = tree.DecisionTreeClassifier()
decision_tree_classifier.fit(train_features_random, train_target)

predictions = decision_tree_classifier.predict(test_features_random)

print("Randomly Chosen Features:", randomly_chosen_labels)
print("Accuracy of Randomly Chosen Features:", accuracy(predictions, test_target))

print("-----------------------------")

# building random forest

# tree 1
predictions_1 = list(make_tree_for_forest(sample_1)) 

# tree 2
predictions_2 = list(make_tree_for_forest(sample_2)) 

# tree 3
predictions_3 = list(make_tree_for_forest(sample_3)) 

# tree 4
predictions_4 = list(make_tree_for_forest(sample_4)) 

# tree 5
predictions_5 = list(make_tree_for_forest(sample_5)) 

predictions = vote(predictions_1, predictions_2, predictions_3, predictions_4, predictions_5)
print("Accuracy Of Random Forest:", accuracy(predictions, test_target))

