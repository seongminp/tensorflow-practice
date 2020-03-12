import pandas as pd
import tensorflow as tf
import os, getopt, sys, json
import numpy as np

MODEL_PATH = 'model.hdf5'
WORD2ENC_PATH = 'word2enc.json'
ENC2WORD_PATH = 'enc2word.json'
META_PATH = 'meta.json'
MAX_NAMES = 20

input_len = {}
names = []
word2enc = {}
enc2word = {}

# Construct encoding / decoding functions.
def encode(text):
    global word2enc
    encoded = []
    words = tf.keras.preprocessing.text.text_to_word_sequence(text)
    for word in words:
        if word in word2enc:
            encoded.append(word2enc[word])
        else:
            encoded.append(0)
    return encoded

def decode(ints):
    global enc2word
    decoded = ""
    PAD = 0
    for int in ints:
        if int != PAD:
            decoded += enc2word[int] + " "
    return decoded[:-1]

def train():
    # Format: Season, Episode, Character, Line
    df = pd.read_csv('All-seasons.csv')

    # Discard rows with invalid values.
    df = df[df[['Season']].apply(lambda x: x['Season'].isdigit(), axis=1)]

    # Convert season and episode columns to numeric data types. 
    df[['Season', 'Episode']] = df[['Season', 'Episode']].apply(pd.to_numeric)

    # Build word index.
    global word2enc
    global enc2word
    word2enc = {}
    enc2word = {}
    encoding = 1
    for index, row in df.iterrows():
        line = tf.keras.preprocessing.text.text_to_word_sequence(row['Line'])
        for word in line:
            if word not in word2enc: 
                word2enc[word] = encoding
                enc2word[encoding] = word
                encoding += 1

    # Save dictionary to disc.
    save_dict(word2enc, WORD2ENC_PATH)
    save_dict(enc2word, ENC2WORD_PATH)

    # Get names of characters with most lines. 
    # Use .index to extract name values from call to nlargest.
    global names
    names = df['Character'].value_counts().nlargest(MAX_NAMES).index.tolist()

    # Extract rows with characters we are concerned about.
    df = df[df[['Character']].apply(lambda x: x['Character'] in names, axis=1)]

    # Get length of longest line.
    global input_len
    input_len = df['Line'].map(lambda x: 
            len(tf.keras.preprocessing.text.text_to_word_sequence(x))).max()

    # Save vocab input len to disk.
    meta = {
            "names": names,
            "input_len": int(input_len)
            }
    save_dict(meta, META_PATH)

    # Convert lines into integer representations.
    lines = df['Line'].apply(lambda x: encode(x))

    # Make all lines of the same length (length of the longest line).
    lines = tf.keras.preprocessing.sequence.pad_sequences(lines.values, input_len)

    # Encode target labels.
    labels = np.array([[names.index(c)] for c in df['Character']])

    # Construct train and test datasets. 
    # lines = features, labels = targets.
    num_samples = len(lines)
    test_samples = int(num_samples * 0.25)
    test_features = lines[0:test_samples]
    test_labels = labels[0:test_samples]
    train_features = lines[test_samples:]
    train_labels = labels[test_samples:]

    # Create model.
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(word2enc)+1, 32),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(MAX_NAMES, activation="softmax")
        ])
    model.summary()

    # Train.
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['acc'])
    history = model.fit(train_features, train_labels, epochs=10)

    # Save model to disk.
    model.save(MODEL_PATH)

    # Evaluate.
    results = model.evaluate(test_features, test_labels)
    return model

def predict(model, text):
    encoded = encode(text)
    encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], input_len)
    pred = np.zeros((1,input_len))
    pred[0] = encoded[0]
    result = model.predict(pred)
    return names[np.argmax(result[0])]

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH) 
    return model

def predict(model, text):
    encoded = encode(text)
    encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], input_len)
    pred = np.zeros((1,input_len))
    pred[0] = encoded[0]
    result = model.predict(pred)
    return names[np.argmax(result[0])]

def save_dict(d, PATH):
    with open(PATH, 'w') as file:
        json.dump(d, file)

def load_word2enc():
    with open(WORD2ENC_PATH, 'r') as file:
        d = json.load(file)
        d = {k: int(v) for k, v in d.items()}
        return d

def load_enc2word():
    with open(ENC2WORD_PATH, 'r') as file:
        d = json.load(file)
        d = {int(k): v for k, v in d.items()}
        return d

def load_meta():
    with open(META_PATH, 'r') as file:
        d = json.load(file)
        return d

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["dry"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        sys.exit(2)

    dry = False
    for o, a in opts:
        if o == '--dry':
            dry = True

    if dry:
        model = train()
    else:
        global word2enc
        global enc2word
        global names
        global input_len
        word2enc = load_word2enc()
        enc2word = load_enc2word()
        meta = load_meta()
        names = meta["names"]
        input_len = int(meta["input_len"])
        model = load_model()

    name = predict(model, "Screw you guys, I'm going home.")
    print(name)

if __name__ == "__main__":
    main()


