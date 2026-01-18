# 🎭 Face Meme Generator

**Real-Time Facial Expression Recognition with Meme Switching**

A fun computer vision mini-project that uses **real-time facial expression recognition** to dynamically display memes based on the user’s facial expression.

The application captures live webcam input, detects the face, predicts the facial expression using a trained deep learning model, and opens **two windows**:

* one showing the live face feed
* another showing a meme that changes according to the detected expression

---

## 🧠 Project Workflow

1. Capture live video using webcam
2. Detect and preprocess the face
3. Predict facial expression using a CNN model
4. Display live face feed in one window
5. Display a meme image corresponding to the predicted expression in a second window

---

## 😄 Supported Expressions

The model predicts the following facial expressions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

Each expression is mapped to a folder of meme images.

---

## 📁 Project Structure

```
FaceMeme/
│── venv/
│── Datasets/
│   │── CK/
│   │── FER/
│   │── RAF/
│
│── meme/
│   │── angry/
│   │── disgust/
│   │── fear/
│   │── happy/
│   │── neutral/
│   │── sad/
│   │── surprise/
│
│── prepare_data.py
│── train_model.py
│── script.py
│── face_meme_model.h5
│── README.md
│── .gitignore
```

---

## 📊 Datasets Used

This project uses a combination of standard facial expression datasets:

* FER (Facial Expression Recognition)
* RAF (Real-world Affective Faces)
* CK (Cohn–Kanade)

> ⚠️ **Datasets are not included in version control** due to size and licensing restrictions.

Place downloaded datasets inside the `Datasets/` directory following the structure shown above.

---

## 🖼️ Meme Mapping

Each predicted expression corresponds to a meme folder:

```
meme/
│── angry/
│── disgust/
│── fear/
│── happy/
│── neutral/
│── sad/
│── surprise/
```

When an expression is detected, a meme from the corresponding folder is displayed in real time.

---

## 🧪 Model Training

* Data is preprocessed using `prepare_data.py`
* Model is trained using `train_model.py`
* Trained model is saved as `face_meme_model.h5`
* The model file is ignored from version control by `.gitignore`

---

## ▶️ How to Run

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

```bash
python prepare_data.py
```

### 4. Train the model (optional)

```bash
python train_model.py
```

### 5. Run the application

```bash
python script.py
```

---

## 🪟 Application Output

* **Window 1:** Live webcam feed with face detection
* **Window 2:** Meme image that updates based on facial expression

---

## 🚫 Ignored Files

The following files and folders are intentionally excluded from GitHub:

* `venv/`
* `Datasets/`
* `face_meme_model.h5`
* Cache and temporary files

---

## 🎯 Purpose of the Project

This project is built for:

* Learning computer vision and deep learning concepts
* Understanding facial expression recognition
* Building interactive ML-based applications
* Academic and portfolio demonstration

---

## ⚠️ Disclaimer

This project is for **educational and experimental purposes only**.
Facial expression predictions may vary depending on lighting, camera quality, and individual facial features.

---

## 🚀 Future Improvements

* Improve model accuracy with better preprocessing
* Add more meme variations per expression
* Optimize real-time performance
* Convert to a GUI-based or desktop application
