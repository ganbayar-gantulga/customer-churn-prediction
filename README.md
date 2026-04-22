# 📊 Customer Churn Prediction

> Харилцагчийн үйлчилгээнээс гарах магадлалыг таамаглах Machine Learning систем

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.1-green?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-yellow)

![Churn Distribution](assets/churn_distribution.png)
![Contract vs Churn](assets/contract_vs_churn.png)
![Monthly Charges and Tenure](assets/charges_tenure.png)
![Correlation Heatmap](assets/correlation_heatmap.png)
![Internet vs Churn](assets/internet_vs_churn.png)
![Model Results](assets/model_results.png)

## 🎯 Төслийн тухай

Энэ төсөл нь харилцаа холбооны компанийн харилцагчдын өгөгдөл дээр суурилан, аль харилцагч үйлчилгээнээс гарах магадлалтайг урьдчилан таамаглах зорилготой. Бизнесийн хувьд шинэ харилцагч олохоос хуучин харилцагчаа авч үлдэх нь 5-7 дахин хямд тул энэ төрлийн загвар практик ач холбогдолтой.

**Гол үр дүн:**
- ✅ XGBoost загвар — **92% accuracy**, **0.89 F1-score**
- ✅ Хамгийн чухал шинж чанаруудыг тодорхойлсон (feature importance)
- ✅ FastAPI ашиглан REST API бүтээсэн
- ✅ Streamlit dashboard ашиглан интерактив UI

## 📁 Төслийн бүтэц

```
customer-churn-prediction/
│
├── README.md                  # Төслийн тайлбар (энэ файл)
├── requirements.txt           # Шаардлагатай сангууд
├── setup.py                   # Package тохиргоо
├── .gitignore                 # Git-ээс хасах файлууд
│
├── data/
│   ├── raw/                   # Анхны түүхий өгөгдөл
│   │   └── telco_churn.csv
│   ├── processed/             # Цэвэрлэсэн өгөгдөл
│   │   ├── train.csv
│   │   └── test.csv
│   └── README.md              # Өгөгдлийн тайлбар
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Өгөгдлийн шинжилгээ (EDA)
│   ├── 02_feature_engineering.ipynb    # Feature боловсруулалт
│   ├── 03_model_training.ipynb        # Модел сургалт & харьцуулалт
│   └── 04_model_evaluation.ipynb      # Загварын үнэлгээ & тайлбар
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py       # Өгөгдөл унших
│   │   └── preprocess.py      # Өгөгдөл цэвэрлэх, хувиргах
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py  # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Модел сургах
│   │   ├── predict.py         # Таамаглал гаргах
│   │   └── evaluate.py        # Модел үнэлэх
│   └── visualization/
│       ├── __init__.py
│       └── plots.py           # Графикуудыг зурах
│
├── models/
│   ├── xgboost_final.pkl      # Хадгалсан загвар
│   └── model_metrics.json     # Загварын үзүүлэлтүүд
│
├── api/
│   ├── main.py                # FastAPI сервер
│   └── schemas.py             # Pydantic моделүүд
│
├── app/
│   └── streamlit_app.py       # Streamlit dashboard
│
├── tests/
│   ├── test_preprocess.py     # Preprocessing тест
│   └── test_model.py          # Model тест
│
├── assets/                    # Зураг, GIF, диаграмм
│   ├── demo.gif
│   ├── eda_charts.png
│   └── feature_importance.png
│
└── Dockerfile                 # Docker тохиргоо (optional)
```

## 🔍 Өгөгдлийн шинжилгээ (EDA)

Telco Customer Churn Dataset ашигласан (7,043 харилцагч, 21 шинж чанар).

| Шинж чанар | Тайлбар |
|-----------|---------|
| `tenure` | Хэдэн сар үйлчлүүлсэн |
| `MonthlyCharges` | Сарын төлбөр |
| `TotalCharges` | Нийт төлбөр |
| `Contract` | Гэрээний төрөл (сар, 1 жил, 2 жил) |
| `InternetService` | Интернэтийн төрөл |
| `Churn` | Гарсан эсэх (Yes/No) — **зорилтот хувьсагч** |

<!-- Энд EDA графикуудын зураг оруулна -->
<!-- ![EDA](assets/eda_charts.png) -->

**Олдсон гол инсайтууд:**
- Сараар гэрээ хийсэн харилцагчид 3 дахин их гардаг
- Эхний 12 сард гарах магадлал хамгийн өндөр
- Fiber optic хэрэглэгчид DSL-ээс илүү их гардаг

## 🤖 Загварууд & Үр дүн

4 загварыг сургаж харьцуулсан:

| Загвар | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.81 | 0.67 | 0.58 | 0.62 | 0.84 |
| Random Forest | 0.85 | 0.73 | 0.65 | 0.69 | 0.87 |
| **XGBoost** | **0.92** | **0.88** | **0.90** | **0.89** | **0.95** |
| LightGBM | 0.90 | 0.85 | 0.87 | 0.86 | 0.93 |

<!-- ![Feature Importance](assets/feature_importance.png) -->

## 🚀 Хэрхэн ажиллуулах

### 1. Клон хийх
```bash
git clone https://github.com/ganbayar0623/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Орчин бэлдэх
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Notebook-ууд ажиллуулах
```bash
jupyter notebook notebooks/
```

### 4. API ажиллуулах
```bash
cd api
uvicorn main:app --reload
```

### 5. Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

## 📦 Шаардлагатай сангууд

```
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
fastapi>=0.108.0
uvicorn>=0.25.0
streamlit>=1.30.0
shap>=0.44.0
joblib>=1.3.0
```

## 📈 Цаашдын хөгжүүлэлт

- [ ] MLflow ашиглан эксперимент хянах
- [ ] Docker-оор containerize хийх
- [ ] CI/CD pipeline нэмэх (GitHub Actions)
- [ ] Cloud-д deploy хийх (AWS/GCP)
- [ ] Өгөгдлийн хэмжээг нэмэгдүүлэх

## 🧠 Сурсан зүйлс

- Imbalanced dataset-тэй ажиллах (SMOTE, class weights)
- Feature engineering-ийн ач холбогдол
- Hyperparameter tuning (Optuna ашиглан)
- SHAP ашиглан модел тайлбарлах (Model Explainability)
- ML моделийг API-аар дамжуулан serve хийх

## 📄 Лиценз

MIT License — Дэлгэрэнгүйг [LICENSE](LICENSE) файлаас харна уу.

## 📬 Холбоо барих

- **Нэр:** Ганбаяр
- **GitHub:** [@ganbayar-gantulga](https://github.com/ganbayar-gantulga))
- **Email:** ganbaayaar0623@gmail.com
- **LinkedIn:** [gantulga-ganbayar](https://www.linkedin.com/in/gantulga-ganbayar-06a391317 ))

---

⭐ Энэ төсөл таалагдсан бол star дарна уу!
