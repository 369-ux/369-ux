- 👋 Hi, I’m @369-ux
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
369-ux/369-ux is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
1. Machine Learning and Deep Learning
Example: Simple Neural Network using TensorFlow

Python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
AI-generated code. Review and use carefully. More info on FAQ.
2. Natural Language Processing (NLP)
Example: Text Classification using BERT

Python

from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Example text
texts = ["This is a positive example.", "This is a negative example."]

# Tokenize the text
inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True)

# Compile the model
model.compile(optimizer=Adam(learning_rate=3e-5), loss=model.compute_loss, metrics=['accuracy'])

# Summary of the model
model.summary()
AI-generated code. Review and use carefully. More info on FAQ.
3. Cognitive Architectures
Example: Simple Cognitive Model using ACT-R (Python)

Python

# Note: ACT-R is typically implemented in Lisp, but here's a conceptual Python example

class CognitiveModel:
    def __init__(self):
        self.memory = {}

    def learn(self, stimulus, response):
        self.memory[stimulus] = response

    def recall(self, stimulus):
        return self.memory.get(stimulus, "No response found")

# Create a cognitive model
model = CognitiveModel()
model.learn("Hello", "Hi there!")
print(model.recall("Hello"))
AI-generated code. Review and use carefully. More info on FAQ.
4. Ethics and Safety
Example: Ethical Decision-Making Framework

Python

def ethical_decision(action, context):
    # Placeholder for ethical decision-making logic
    if action == "harm" and context == "human":
        return "Action not allowed"
    return "Action allowed"

# Example usage
action = "harm"
context = "human"
print(ethical_decision(action, context))
1. Enhanced Simple Neural Network using TensorFlow
Let's add a training phase and validation data to the neural network example.

python
Copy code
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0  # Flatten and normalize
x_test = x_test.reshape(-1, 784) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Create a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Summary of the model
model.summary()
2. Enhanced Text Classification using BERT
We'll add functionality for predictions on new texts.

python
Copy code
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Example texts
texts = ["This is a positive example.", "This is a negative example."]

# Tokenize the texts
inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True)

# Compile the model
model.compile(optimizer=Adam(learning_rate=3e-5), loss=model.compute_loss, metrics=['accuracy'])

# Train the model (add dummy labels for demonstration)
labels = np.array([1, 0])  # Assume 1 for positive, 0 for negative
model.fit(inputs, labels, epochs=3)

# Function to classify new texts
def classify_text(new_texts):
    inputs = tokenizer(new_texts, return_tensors='tf', padding=True, truncation=True)
    logits = model(inputs).logits
    predictions = tf.argmax(logits, axis=-1)
    return predictions.numpy()

# Example usage
new_texts = ["I love this!", "This is terrible."]
predictions = classify_text(new_texts)
print(predictions)  # Outputs: array of predicted labels
3. Enhanced Cognitive Model using ACT-R Concept
Let’s expand the cognitive model to include more sophisticated learning and recall methods.

python
Copy code
class CognitiveModel:
    def __init__(self):
        self.memory = {}

    def learn(self, stimulus, response):
        self.memory[stimulus] = response
        print(f"Learned: {stimulus} -> {response}")

    def recall(self, stimulus):
        response = self.memory.get(stimulus)
        if response:
            return response
        else:
            return "No response found"

    def list_memory(self):
        return self.memory

# Create a cognitive model
model = CognitiveModel()
model.learn("Hello", "Hi there!")
model.learn("Goodbye", "See you later!")
print(model.recall("Hello"))  # Should return "Hi there!"
print(model.recall("Unknown"))  # Should return "No response found"
print(model.list_memory())  # Outputs the entire memory
4. Enhanced Ethical Decision-Making Framework
We can expand the ethical decision-making framework to include more complex conditions.

python
Copy code
def ethical_decision(action, context, stakeholders):
    # Placeholder for ethical decision-making logic
    if action == "harm" and context == "human":
        return "Action not allowed due to potential harm to humans."
    
    if action == "lie" and "trust" in stakeholders:
        return "Action not allowed; trust is paramount."

    return "Action allowed"

# Example usage
print(ethical_decision("harm", "human", []))  # Outputs: Action not allowed due to potential harm to humans.
print(ethical_decision("lie", "social", ["trust"]))  # Outputs: Action not allowed; trust is paramount.
print(ethical_decision("help", "human", []))  # Outputs: Action allowed.
These enhancements provide more functionality and context for each example. If you have further specific requirements or questions, feel free to let me know!


I prefer this response
ChatGPT
Response 2
Sure! Let’s enhance each code example with additional functionality, comments, and best practices.

1. Improved Simple Neural Network using TensorFlow
We'll add a training phase, validation split, and some metrics for better performance evaluation.

python
Copy code
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255  # Flatten and normalize
x_test = x_test.reshape(-1, 784).astype('float32') / 255

# Create a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation data
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Summary of the model
model.summary()
2. Enhanced Text Classification using BERT
Let’s add a function for inference and expand the example with better handling for the output.

python
Copy code
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Function for text classification
def classify_text(texts):
    inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True)
    outputs = model(inputs)
    predictions = np.argmax(outputs.logits.numpy(), axis=-1)
    return predictions

# Example text
texts = ["This is a positive example.", "This is a negative example."]
predictions = classify_text(texts)
print("Predictions:", predictions)

# Compile the model (if training)
model.compile(optimizer=Adam(learning_rate=3e-5), loss=model.compute_loss, metrics=['accuracy'])

# Summary of the model
model.summary()
3. Enhanced Cognitive Model using ACT-R Concept
We can add more functionality for learning and recalling multiple responses, as well as a simple interface for interaction.

python
Copy code
class CognitiveModel:
    def __init__(self):
        self.memory = {}

    def learn(self, stimulus, response):
        self.memory[stimulus] = response
        print(f"Learned: {stimulus} -> {response}")

    def recall(self, stimulus):
        response = self.memory.get(stimulus, "No response found")
        return response

    def learn_multiple(self, pairs):
        for stimulus, response in pairs:
            self.learn(stimulus, response)

# Create a cognitive model
model = CognitiveModel()
model.learn_multiple([
    ("Hello", "Hi there!"),
    ("How are you?", "I'm a machine, but thanks for asking!")
])

# Example recall
print(model.recall("Hello"))
print(model.recall("What is your name?"))
4. Improved Ethical Decision-Making Framework
We’ll add more complexity by allowing multiple contexts and refining the decision-making logic.

python
Copy code
def ethical_decision(action, context):
    ethical_standards = {
        "harm": ["human", "animal"],
        "deceive": ["human"],
        "help": ["human", "animal"]
    }
    
    if action in ethical_standards and context in ethical_standards[action]:
        return "Action not allowed"
    
    return "Action allowed"

# Example usage
actions = [("harm", "human"), ("deceive", "human"), ("help", "animal")]
for action, context in actions:
    print(f"Action: {action}, Context: {context} -> {ethical_decis
1. Enhanced Simple Neural Network using TensorFlow
We’ll add a training phase, validation split, and some metrics for better performance evaluation.

Python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0  # Flatten and normalize
x_test = x_test.reshape(-1, 784) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Create a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation data
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Summary of the model
model.summary()
AI-generated code. Review and use carefully. More info on FAQ.
2. Enhanced Text Classification using BERT
We’ll add functionality for predictions on new texts.

Python

from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Example texts
texts = ["This is a positive example.", "This is a negative example."]

# Tokenize the texts
inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True)

# Compile the model
model.compile(optimizer=Adam(learning_rate=3e-5), loss=model.compute_loss, metrics=['accuracy'])

# Train the model (add dummy labels for demonstration)
labels = np.array([1, 0])  # Assume 1 for positive, 0 for negative
model.fit(inputs, labels, epochs=3)

# Function to classify new texts
def classify_text(new_texts):
    inputs = tokenizer(new_texts, return_tensors='tf', padding=True, truncation=True)
    logits = model(inputs).logits
    predictions = tf.argmax(logits, axis=-1)
    return predictions.numpy()

# Example usage
new_texts = ["I love this!", "This is terrible."]
predictions = classify_text(new_texts)
print(predictions)  # Outputs: array of predicted labels
AI-generated code. Review and use carefully. More info on FAQ.
3. Enhanced Cognitive Model using ACT-R Concept
Let’s expand the cognitive model to include more sophisticated learning and recall methods.

Python

class CognitiveModel:
    def __init__(self):
        self.memory = {}

    def learn(self, stimulus, response):
        self.memory[stimulus] = response
        print(f"Learned: {stimulus} -> {response}")

    def recall(self, stimulus):
        response = self.memory.get(stimulus)
        if response:
            return response
        else:
            return "No response found"

    def list_memory(self):
        return self.memory

# Create a cognitive model
model = CognitiveModel()
model.learn("Hello", "Hi there!")
model.learn("Goodbye", "See you later!")
print(model.recall("Hello"))  # Should return "Hi there!"
print(model.recall("Unknown"))  # Should return "No response found"
print(model.list_memory())  # Outputs the entire memory
AI-generated code. Review and use carefully. More info on FAQ.
4. Enhanced Ethical Decision-Making Framework
We can expand the ethical decision-making framework to include more complex conditions.

Python

def ethical_decision(action, context, stakeholders):
    # Placeholder for ethical decision-making logic
    if action == "harm" and context == "human":
        return "Action not allowed due to potential harm to humans."
    
    if action == "lie" and "trust" in stakeholders:
        return "Action not allowed; trust is paramount."

    return "Action allowed"

# Example usage
print(ethical_decision("harm", "human", []))  # Outputs: Action not allowed due to potential harm to humans.
print(ethical_decision("lie", "social", ["trust"]))  # Outputs: Action not allowed; trust is paramount.
print(ethical_decision("help", "human", []))  # Outputs: Action allowed.
