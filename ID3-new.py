import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import math

# Load your dataset from the CSV file
file_path = "updated_dataset.csv"  # Replace with the actual file path
data = pd.read_csv(file_path)

# List of attributes (excluding the target attribute)
attributes = ['Mood', 'Effort', 'Score']

# Target attribute
target_attribute = 'Output'

# Function to calculate entropy
def entropy(data, target_attribute):
    classes = data[target_attribute].unique()
    entropy_value = 0
    total_examples = len(data)

    for c in classes:
        class_prob = len(data[data[target_attribute] == c]) / total_examples
        entropy_value -= class_prob * math.log2(class_prob)

    return entropy_value

# Function to calculate information gain
def information_gain(data, attribute, target_attribute):
    entropy_before_split = entropy(data, target_attribute)
    values = data[attribute].unique()

    entropy_after_split = 0
    total_examples = len(data)

    for value in values:
        subset_data = data[data[attribute] == value]
        subset_entropy = entropy(subset_data, target_attribute)
        subset_prob = len(subset_data) / total_examples
        entropy_after_split += subset_prob * subset_entropy

    return entropy_before_split - entropy_after_split

# Function to find the best attribute
def find_best_attribute(data, attributes, target_attribute):
    best_attribute = None
    max_gain = -1

    for attribute in attributes:
        gain = information_gain(data, attribute, target_attribute)
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute

    return best_attribute, max_gain

# Function to build the decision tree
def my_ID3(data, attributes, target_attribute):
    if len(set(data[target_attribute])) == 1:
        return data[target_attribute].iloc[0]

    if len(attributes) == 0:
        majority_class = data[target_attribute].mode()[0]
        return majority_class

    best_attribute, gain = find_best_attribute(data, attributes, target_attribute)
    print(f"Attribute {best_attribute} with Gain = {gain} is chosen as the decision attribute")

    # Create a column transformer for one-hot encoding categorical attributes
    categorical_features = data[attributes].select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create a pipeline with preprocessing and decision tree
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
    ])

    # Fit the pipeline to the data
    pipeline.fit(data[attributes], data[target_attribute])

    # Plot the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(pipeline['classifier'], feature_names=pipeline.named_steps['preprocessor'].get_feature_names_out(attributes),
              class_names=data[target_attribute].unique(), filled=True, rounded=True)
    plt.savefig("decision_tree.png")

    return pipeline

# Build the decision tree
my_tree = my_ID3(data, attributes, target_attribute)

# Display the decision tree
plt.show()
