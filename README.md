# END-TO-END-DATA-SCIENCE

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MANIGANDAN

*INTERN ID*: CT04DN1091

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

# 🌸 Iris Flower Classification API (Flask)

This project demonstrates an end-to-end machine learning pipeline using the **Iris flower dataset**, deployed via a **Flask API**.

## 📌 Project Features
- Trains a Random Forest model on the Iris dataset
- Saves the model as a `.pkl` file using `joblib`
- Builds a Flask API with a `/predict` endpoint
- Accepts JSON input and returns the predicted flower species

---

## 📁 Files in this Project
| File           | Description                                      |
|----------------|--------------------------------------------------|
| `train_model.py` | Trains and saves the model as `iris_model.pkl` |
| `iris_model.pkl` | The saved trained model file                    |
| `app.py`         | Flask API code with prediction endpoint         |
| `test_api.py`    | Optional script to test the API locally         |
| `README.md`      | You're reading it!                              |

---

## 🚀 How to Run Locally

1. Install dependencies
'bash
pip install flask scikit-learn joblib

2.Train the model
python train_model.py

3. Run the Flask API
python app.py

4.Flask server will start at:
http://127.0.0.1:5000


## 📫 Contact

Made with ❤️ by Manigandan



