# Cost Compass 🧭
> Navigating the True Price of Living Through Data

A modular Python backend that estimates the **real monthly cost of living** in cities worldwide. It merges and cleans two Kaggle datasets, computes five expense categories (bills, transport, groceries, entertainment, accommodation), and predicts whether a typical resident would live in **comfort or budget mode** using a trained Random Forest classifier — all exposed through a Flask REST API.

**Academic project — ENSAM Rabat, Python for Data Science & ML, 2025/2026**  
**Presented:** 17 March 2026

---

## Team

| Name | Contribution |
|------|-------------|
| **Yasmine Lahlaoi** | Backend — data processing, ML pipeline |
| **Alae Mouhssine** | Backend — data processing, ML pipeline |
| **Kawtar Khallady** | Frontend — React + Redux dashboard *(not included in this repository)* |

> The frontend (React + Redux Toolkit) is maintained in a separate repository. It connects to this API and renders an interactive dashboard with animated charts, a donut expense breakdown, and accommodation comparisons.

---

## What It Does

You give it a city name. It returns:
- Monthly net salary for that city
- Itemised costs for bills, transport, groceries, entertainment, and accommodation
- A **comfort / budget mode** prediction (Random Forest classifier)
- In budget mode: all variable expenses are automatically recalculated using the cheapest available options
- Remaining income after all expenses

---

## Project Structure

```
cost-compass/
│
├── data/                        
│   ├── cost-of-living.csv
│   └── cost-of-living_v2.csv
│
├── models/                      # ← auto-created by train_mode.py
│   ├── mode_model.pkl
│   ├── mode_importance.png
│   └── Mode model: RandomForestClassifier_report.txt
│
├── data_preprocessing.py        # Data cleaning + all expense computation functions
├── train_mode.py                # ML training pipeline (Random Forest + GridSearchCV)
├── main.py                      # Core orchestrator — produces the full city dashboard
├── app.py                       # Flask REST API
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & How to Run

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/cost-compass.git
cd cost-compass
```

### 2. Create a virtual environment & install dependencies
```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Train the ML model *(run once before starting the API)*
```bash
python train_mode.py
```
This reads both CSVs, engineers features, runs GridSearchCV, and writes:
- `models/mode_model.pkl` — the trained classifier
- `models/mode_importance.png` — feature importance chart
- `models/Mode model: RandomForestClassifier_report.txt` — accuracy/F1 report

Training takes a few minutes due to the GridSearchCV hyperparameter grid.

> **Skip this step?** The API still works without the model — it falls back to a simple rule: budget if expenses > salary.

### 4. Start the API
```bash
python app.py
```
The server starts at `http://localhost:5000`.

### 5. Query a city from the command line *(no server needed)*
```bash
python main.py Paris
python main.py "New York"
python main.py Tokyo
```

---

## API Reference

### `GET /city/<city_name>`
Returns a full cost-of-living breakdown for the requested city.

**Example:**
```bash
curl http://localhost:5000/city/Paris
```

**Response:**
```json
{
  "city": "Paris",
  "country": "France",
  "mode": "comfort",
  "net_salary":     { "monthly_value": 2800.0 },
  "bills_and_fees": {
    "monthly_average_cost": 180.5,
    "monthly_details": {
      "monthly_electricity_heating_water_cost": 120.0,
      "monthly_mobile": 36.0,
      "monthly_internet_cost": 24.5
    }
  },
  "transportation": {
    "monthly_average_cost": 320.0,
    "monthly_details": {
      "monthly_public_transport_cost": 85.0,
      "monthly_gasoline_cost": 320.0,
      "transport_type": "car"
    }
  },
  "groceries":      { "monthly_average_cost": 420.0, "monthly_details": {} },
  "entertainment":  { "monthly_total_cost":   210.0, "monthly_details": {} },
  "accommodations": { "monthly_total_cost":  1100.0, "monthly_details": {} },
  "summary": {
    "total_monthly_expenses": 2230.5,
    "remaining_income": 569.5
  },
  "error": ""
}
```

**Error response (city not found):**
```json
{ "mode": "error", "error": "City 'Xyz' not found in the dataset." }
```

---


**Request body:**
```json
{
  "salary":           3000,
  "location":         "center",
  "housing_type":     "studio",
  "transport_type":   "public",
  "gas_liters":       0,
  "cheap_visits":     4,
  "mid_visits":       2,
  "fast_food_visits": 2,
  "go_gym":           true,
  "play_tennis":      false,
  "tennis_frequency": 0,
  "go_cinema":        true,
  "cinema_visits":    2
}
```

**Response:** array of 5 city objects ranked by remaining income after expenses.

---

## How It Works

### Expense Categories

| Category | Logic |
|----------|-------|
| **Bills** | Electricity + water + 60 min/day mobile + internet |
| **Transport** | Compares public transit (monthly pass or 40 tickets + 10 taxi rides) vs car (80 L fuel/month). Comfort = more expensive option; budget = cheaper option. |
| **Groceries** | Fixed monthly basket: 8 L milk, 2 kg rice, 4 doz eggs, 10 loaves bread, 1 kg cheese, 4 kg chicken, 2 kg beef, fruits & veg × 4 weeks |
| **Entertainment** | Comfort: gym × 3 + tennis × 8 + cinema × 2 + restaurant × 4 + fast food × 2. Budget: cheapest sport + cheapest food option + cinema. |
| **Accommodation** | Comfort: average of 4 rent options (studio/family × centre/outside). Budget: cheapest available rent. |

### Comfort vs Budget Mode

The mode is predicted by a **Random Forest classifier** trained on the five expense totals above. If expenses are sustainable on the city's average salary the model predicts `comfort`; otherwise `budget`. In budget mode all variable categories are automatically recalculated using their cheapest options.

| Metric | Value |
|--------|-------|
| Accuracy  | 0.862 |
| Precision | 0.861 |
| Recall    | 0.863 |
| F1-score  | 0.862 |

If the model file is not found the app falls back silently to: budget if `total_expenses > salary`.

### Missing Value Strategy

1. **Country-level median** — a missing price in Paris is estimated from other French cities.
2. **Global column median** — fallback for countries with only one city and no data.

Median is preferred over mean to avoid distortion from extreme outliers.

---

## Dataset

Both CSV files are included in the `data/` folder. They are sourced from the
[Global Cost of Living — Kaggle](https://www.kaggle.com/datasets/mvieira101/global-cost-of-living/data)
dataset by Miguel Piedade. Note that prices reflect the time of data collection
and may not match current figures.

---

## Known Limitations

- Salary data (`x54`) is the **city average** — individual salaries vary widely
- Grocery basket and activity frequencies assume a single adult
- No healthcare, insurance, taxes, or education costs included
- Dataset prices reflect the time of collection, not today's prices
- Comfort mode always picks the more expensive transport option as a proxy for higher spending — may not match every user's real preference

---

## License
Built for academic purposes at ENSAM Rabat. Feel free to use it as a reference for your own data science projects.
