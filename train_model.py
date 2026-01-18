import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 7

# Paths
BASE_DIR = r"d:\Comdur\Main Projects\FaceMeme\Datasets"
FER_DIR = os.path.join(BASE_DIR, "FER", "train")
RAF_DIR = os.path.join(BASE_DIR, "RAF", "train")
RAF_LABELS_PATH = os.path.join(BASE_DIR, "RAF", "train_labels.csv")
MODEL_SAVE_PATH = r"d:\Comdur\Main Projects\FaceMeme\face_meme_model.h5"

# Mappings
# Common keys: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# DeepFace/Standard Index: 0:angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
# Wait, let's stick to alphabetical order for consistency with Keras `flow_from_directory`:
# ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# RAF Label Parsing
# RAF-DB: 1: Surprise, 2: Fear, 3: Disgust, 4: Happy, 5: Sad, 6: Angry, 7: Neutral
RAF_MAP = {
    1: 'surprise',
    2: 'fear',
    3: 'disgust',
    4: 'happy',
    5: 'sad',
    6: 'angry',
    7: 'neutral'
}

def load_raf_data():
    if not os.path.exists(RAF_LABELS_PATH):
        print("RAF labels not found, skipping RAF.")
        return []

    print("Loading RAF labels...")
    df = pd.read_csv(RAF_LABELS_PATH)
    # columns: image, label
    
    data = []
    for _, row in df.iterrows():
        fname = row['image']
        label_id = int(row['label'])
        
        # fix filename if needed (sometimes 'train_00001_aligned.jpg')
        # user file listing showed 'train_00001_aligned.jpg', which matches mostly
        
        full_path = os.path.join(RAF_DIR, fname)
        if not os.path.exists(full_path):
            # Try removing '_aligned' if not found? Or assume it's correct
            continue
            
        emotion = RAF_MAP.get(label_id)
        if emotion:
            data.append((full_path, CLASS_TO_IDX[emotion]))
            
    return data

def load_fer_data():
    if not os.path.exists(FER_DIR):
        print("FER dir not found, skipping.")
        return []

    print("Loading FER file paths...")
    data = []
    for cls in CLASSES:
        cls_dir = os.path.join(FER_DIR, cls)
        if os.path.exists(cls_dir):
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    data.append((os.path.join(cls_dir, f), CLASS_TO_IDX[cls]))
    return data

class UnifiedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_list, batch_size=32, dim=(224, 224), n_channels=3, n_classes=7, shuffle=True):
        self.data_list = data_list
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.data_list[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, (path, label) in enumerate(list_IDs_temp):
            # Read image
            img = cv2.imread(path)
            if img is None:
                # Should handle this better, but filling with zeros for now to avoid crash
                img = np.zeros((*self.dim, self.n_channels))
            else:
               # Resize
               img = cv2.resize(img, self.dim)
               # RGB
               img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
               
            # Normalize
            X[i,] = img / 255.0
            y[i] = label

        return X, to_categorical(y, num_classes=self.n_classes)

def create_model():
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), 
                             include_top=False, 
                             weights='imagenet')
    
    # Freeze base model first
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    # 1. Prepare Data
    raf_data = load_raf_data()
    fer_data = load_fer_data()
    
    all_data = raf_data + fer_data
    print(f"Total images found: {len(all_data)} (RAF: {len(raf_data)}, FER: {len(fer_data)})")
    
    if len(all_data) == 0:
        print("No data found! Check paths.")
        return

    # Shuffle once globally
    np.random.shuffle(all_data)
    
    # Simple split
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    train_gen = UnifiedDataGenerator(train_data, batch_size=BATCH_SIZE, dim=(IMG_SIZE, IMG_SIZE))
    val_gen = UnifiedDataGenerator(val_data, batch_size=BATCH_SIZE, dim=(IMG_SIZE, IMG_SIZE))
    
    # 2. Build Model
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # 3. Train
    print("Starting training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )
    
    # 4. Save
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
