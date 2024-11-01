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
