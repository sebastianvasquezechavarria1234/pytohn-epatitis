# 🔬 Hepatitis Prediction API

Professional medical outcome prediction system based on Machine Learning. This project provides a robust API for predicting hepatitis patient survival ("Vive" or "Muere") based on 21 clinical features.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 🚀 Key Features

- **Modern Architecture**: Built with **FastAPI** for high performance and asynchronous support.
- **Strict Validation**: Utilizes **Pydantic** to ensure data integrity for all 21 medical parameters.
- **Automatic Documentation**: Interactive Swagger UI available at `/docs`.
- **Professional Logging**: Structured request monitoring and global error handling.
- **Dynamic Metadata**: Decoupled model info for easier maintenance and transparency.

## 🛠️ Tech Stack

- **Core**: Python 3.x
- **Framework**: FastAPI
- **ML Engine**: Scikit-Learn (Logistic Regression)
- **Validation**: Pydantic
- **Server**: Uvicorn

## 📋 Medical Features (Inputs)

The model evaluates **21 clinical parameters**, including:
- **Demographics**: Age, Sex, Marital Status, City.
- **Symptoms**: Fatigue, Malaise, Anorexia.
- **Clinical Signs**: Steroid use, Antivirals, Liver size/firmness, Spleen palpable, Spiders, Ascites, Varices.
- **Lab Results**: Bilirubin, Alk Phosphate, Sgot, Albumin, Protime, Histology.

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sebastianvasquezechavarria1234/pytohn-epatitis.git
   cd pytohn-epatitis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```bash
   python main.py
   ```
   *The server will start at `http://localhost:5000`*

## 📖 API Documentation

Once the server is running, you can access:
- **Interactive Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

### Example Request (`POST /predict`)

```json
{
  "Age": 30.0,
  "Sex": 1,
  "Estado_Civil": 1,
  "Ciudad": 1,
  "Steroid": 1,
  "Antivirals": 2,
  "Fatigue": 1,
  "Malaise": 1,
  "Anorexia": 1,
  "Liver_Big": 2,
  "Liver_Firm": 1,
  "Spleen_Palpable": 1,
  "Spiders": 1,
  "Ascites": 1,
  "Varices": 1,
  "Bilirubin": 1.0,
  "Alk_Phosphate": 85.0,
  "Sgot": 18.0,
  "Albumin": 4.0,
  "Protime": 100.0,
  "Histology": 1
}
```

---
*Disclaimer: This tool is for educational and research purposes. Medical decisions should always be made by qualified healthcare professionals.*
