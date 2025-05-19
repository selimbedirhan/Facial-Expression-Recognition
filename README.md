```markdown
# Facial Expression Recognition with PyTorch 🤖🙂🙁😲

This project uses a **ResNet-18** deep learning model with transfer learning to perform facial expression recognition from images and live webcam feed.

---

## 📁 Project Structure

- `facial_expression_recognition.py`: Trains a ResNet-18 model on a facial expression dataset.
- `test.py`: Loads the saved model weights and verifies successful loading.
- `predict_image.py`: (Optional) Predicts the facial expression from a single image.
- `camera_test.py`: Runs real-time expression recognition using your webcam.
- `resnet18_facial_expression.pth`: Trained model weights (generated after training).
- `dataset/`: Folder where you should place your dataset. It must be in **ImageFolder** structure:
```

dataset/
├── train/
│   ├── Angry/
│   ├── Happy/
│   └── ...
└── test/
├── Angry/
├── Happy/
└── ...

````

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Torchvision
- OpenCV (`cv2`)
- Pillow (`PIL`)
- numpy

Install requirements:

```bash
pip install torch torchvision opencv-python pillow numpy
````

---

## 🚀 Training

To train the model:

```bash
python facial_expression_recognition.py
```

This will:

* Load data from `dataset/train` and `dataset/test`
* Fine-tune a pre-trained ResNet-18 model
* Save the weights as `resnet18_facial_expression.pth`

---

## 🧪 Testing (Model Check)

To verify model loading:

```bash
python test.py
```

---

## 📷 Real-Time Prediction (Webcam)

Make sure `resnet18_facial_expression.pth` exists, then run:

```bash
python camera_test.py
```

Press **`q`** to quit the webcam window.

---

## 🖼️ Predict from a Single Image

Coming soon in `predict_image.py` 👀
You will be able to use:

```bash
python predict_image.py path_to_image.jpg
```

> Make sure to update `class_names` according to your dataset's class folders.

---

## 🧠 Pretrained Model Info

* Base model: **ResNet-18**
* Final layer modified to match the number of expression classes (e.g., 7: Angry, Happy, Sad, etc.)
* Transfer learning enabled: only the final layer is trained.

---

## 📊 Dataset Suggestions

You can use datasets like:

* [FER-2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
* Your own collected image dataset, following ImageFolder structure.

---

## 📌 License

MIT License

---

## 👤 Author

**Selim Bedirhan Öztürk**

> Developed as part of a deep learning study and facial emotion recognition exploration.

