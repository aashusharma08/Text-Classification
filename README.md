# Text Emotion Classification Model

This project focuses on building a text emotion classification model using Keras and TensorFlow. The model classifies text data into various emotional categories based on the input text.

## Requirements

To run the code in this repository, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- scikit-learn

You can install the required libraries by running:

```bash
pip install tensorflow keras pandas numpy scikit-learn
```

## Dataset

The model uses a text dataset stored in a file called `train.txt`. This file contains text data with emotions. Each line consists of a text sample and its associated emotion label, separated by a semicolon (`;`).

Example data format in `train.txt`:

```
i can go from feeling so hopeless to so damned...; sadness
im grabbing a minute to post i feel greedy wrong; anger
i am ever feeling nostalgic about the fireplace...; love
```

- `Text`: The text data (sentences).
- `Emotions`: The corresponding emotion label (e.g., anger, sadness, joy, etc.).

## Steps

### 1. Data Loading and Preprocessing
- The data is loaded from a CSV file using `pandas`.
- The text data is tokenized using Keras's `Tokenizer` class.
- Sequences are padded to ensure consistent length for input into the model.
- The emotion labels are encoded as integers using `LabelEncoder` and then one-hot encoded.

### 2. Model Architecture
- A Sequential neural network is used for classification.
  - **Embedding Layer**: Converts words into dense vectors of fixed size.
  - **Flatten Layer**: Flattens the 2D input from the embedding into a 1D array.
  - **Dense Layers**: Fully connected layers for classification.

- The model uses the **Adam optimizer** and **categorical cross-entropy loss** function.

### 3. Model Training
- The data is split into training and testing sets using `train_test_split`.
- The model is trained for 15 epochs with a batch size of 128.

### 4. Evaluation
- The model's performance is evaluated using accuracy on the test data.
- Validation accuracy improves gradually through the epochs.

### 5. Prediction
- After training, you can input a text sentence, and the model will predict the emotion label.
  
Example of input:

```python
input_text = "He is admit in hospital today, so he wouldn't be able to show his presence today in the meeting."
```

The output will be the predicted emotion:

```
['fear']
```

## How to Use

1. Prepare your own dataset in the format described above (`Text; Emotion`).
2. Update the `train.txt` file with the new data.
3. Run the script, and the model will process the data and train itself.
4. Use the trained model to predict emotions for new sentences.

```python
input_text = "Your new sentence here."
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
prediction = model.predict(padded_input_sequence)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
print(predicted_label)
```

## Potential Improvements
- You can experiment with different neural network architectures.
- Pre-trained word embeddings (e.g., GloVe, Word2Vec) can be used instead of training embeddings from scratch.
- Hyperparameter tuning can improve the model's performance.
- Data augmentation techniques can be applied to enhance the dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
