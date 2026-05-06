"""
data_preprocessing.py
----------------------
ML-enhanced backend for the Cost-of-Living dashboard.

Responsibilities
----------------
- preprocess_data        : load & clean both CSV sources
- compute_net_salary     : extract monthly net salary
- compute_bills          : utilities + mobile + internet
- compute_transportation : public transport vs car (budget/comfort)
- compute_groceries      : monthly grocery basket
- compute_entertainment  : leisure & dining (budget/comfort)
- compute_accommodations : rent scenarios (budget/comfort)
- compute_summary        : aggregate totals; ML model decides the mode
"""

import os
import joblib
import numpy as np
import pandas as pd


# ── Load ML model once at startup ────────────────────────────────────────────
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
try:
    _mode_model = joblib.load(os.path.join(_MODEL_DIR, "mode_model.pkl"))
    _ML_READY = True
except Exception:
    _ML_READY = False  # app still works before the model is trained


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_data(file_path1: str, file_path2: str):
    """
    Load and merge two cost-of-living CSV files.

    Missing values are filled in two passes:
      1. Country-level median  (e.g. a missing Paris price uses the French median).
      2. Global column median as a fallback for countries with no data at all.

    Median is preferred over mean to limit the influence of extreme outliers.
    Cities with no salary data (x54 == 0) are removed.

    Returns
    -------
    (df, error_message)  — error_message is None on success.
    """
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)

        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop_duplicates(subset=["city", "country"], keep="first")
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")

        cols = [
            "x54", "x36", "x37", "x38", "x28", "x29", "x30", "x31", "x33",
            "x9",  "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17",
            "x18", "x19", "x20", "x21", "x22", "x1",  "x2",  "x3",  "x39",
            "x40", "x41", "x48", "x49", "x50", "x51",
        ]

        # Pass 1: country-level median
        df[cols] = df.groupby("country")[cols].transform(
            lambda x: x.fillna(x.median())
        )
        # Pass 2: global median fallback
        df[cols] = df[cols].fillna(df[cols].median())

        df = df[df["x54"] != 0]
        return df, None

    except FileNotFoundError as e:
        return {}, f"Error: One or both CSV files not found. {e}"
    except pd.errors.EmptyDataError as e:
        return {}, f"Error: One or both CSV files are empty. {e}"
    except Exception as e:
        return {}, f"Unexpected error in preprocess_data: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. NET SALARY
# ══════════════════════════════════════════════════════════════════════════════

def compute_net_salary(row):
    """Return the monthly net salary from column x54."""
    try:
        net_salary = {"monthly_value": row["x54"]}
        return net_salary, net_salary["monthly_value"], None
    except KeyError as e:
        return {}, 0, f"Error: Missing column {e} in compute_net_salary."
    except Exception as e:
        return {}, 0, f"Unexpected error in compute_net_salary: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. BILLS
# ══════════════════════════════════════════════════════════════════════════════

def compute_bills(row):
    """
    Estimate monthly utility bills.

    - Electricity / heating / water  : x36
    - Mobile (60 min/month)          : x37 × 60
    - Internet                       : x38
    """
    try:
        bills_details = {
            "monthly_electricity_heating_water_cost": row["x36"],
            "monthly_mobile":                         round(row["x37"] * 60, 2),
            "monthly_internet_cost":                  row["x38"],
        }
        bills_avg = round(sum(bills_details.values()), 2)
        bills_and_fees = {
            "monthly_average_cost": bills_avg,
            "monthly_details":      bills_details,
        }
        return bills_and_fees, bills_avg, None

    except KeyError as e:
        return {}, 0, f"Error: Missing column {e} in compute_bills."
    except Exception as e:
        return {}, 0, f"Unexpected error in compute_bills: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRANSPORTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_transportation(row, budget_mode: bool = False):
    """
    Compare two monthly transport scenarios.

    Scenario A – Public transport
        • Individual tickets: x28 × 40  (2 trips/day, 5 days/week)
        • Monthly pass:       x29        (used if cheaper than individual tickets)
        • Occasional taxi:   (x30 + x31) × 10

    Scenario B – Own car
        • Gasoline: x33 × 80 litres/month

    Budget mode  → cheapest option.
    Comfort mode → more expensive option (assumes the user is already
                   comfortable spending the higher amount).
    """
    try:
        individual_tickets = row["x28"] * 40
        monthly_pass       = row["x29"]
        taxi_cost          = (row["x30"] + row["x31"]) * 10

        public_base = (
            monthly_pass if monthly_pass > 0 and monthly_pass < individual_tickets
            else individual_tickets
        )
        public_transport_total = round(public_base + taxi_cost, 2)
        car_total              = round(row["x33"] * 80, 2)

        cheapest_cost  = min(public_transport_total, car_total)
        expensive_cost = max(public_transport_total, car_total)

        if budget_mode:
            transport_total = cheapest_cost
            transport_type  = "public" if public_transport_total <= car_total else "car"
        else:
            transport_total = expensive_cost
            transport_type  = "public" if public_transport_total >= car_total else "car"

        transport_details = {
            "monthly_public_transport_cost": public_transport_total,
            "monthly_gasoline_cost":         car_total,
            "transport_type":                transport_type,
        }
        transportation = {
            "monthly_average_cost": transport_total,
            "monthly_details":      transport_details,
        }
        return transportation, transport_total, cheapest_cost, None

    except KeyError as e:
        return {}, 0, 0, f"Error: Missing column {e} in compute_transportation."
    except Exception as e:
        return {}, 0, 0, f"Unexpected error in compute_transportation: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. GROCERIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_groceries(row):
    """
    Estimate monthly grocery spend for a single person.

    Quantities per month:
        Milk 8 L · Rice 2 kg · Eggs 4 doz · Bread 10 loaves · Cheese 1 kg
        Chicken 4 kg · Beef 2 kg · Fruits & vegetables (7 items) × 4 weeks
    """
    try:
        grocery_items = {
            "milk":           round(row["x9"]  * 8,  2),
            "rice":           round(row["x11"] * 2,  2),
            "eggs":           round(row["x12"] * 4,  2),
            "bread":          round(row["x10"] * 10, 2),
            "cheese":         round(row["x13"] * 1,  2),
            "chicken":        round(row["x14"] * 4,  2),
            "beef":           round(row["x15"] * 2,  2),
            "fruits_veggies": round(
                (row["x16"] + row["x17"] + row["x18"] + row["x19"] +
                 row["x20"] + row["x21"] + row["x22"]) * 4, 2
            ),
        }
        groceries_avg = sum(grocery_items.values())
        groceries = {
            "monthly_average_cost": groceries_avg,
            "monthly_details":      grocery_items,
        }
        return groceries, groceries_avg, None

    except KeyError as e:
        return {}, 0, f"Error: Missing column {e} in compute_groceries."
    except Exception as e:
        return {}, 0, f"Unexpected error in compute_groceries: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. ENTERTAINMENT
# ══════════════════════════════════════════════════════════════════════════════

def compute_entertainment(row, budget_mode: bool = False):
    """
    Estimate monthly leisure & dining costs.

    Comfort mode → full basket (tennis + gym + cinema + restaurant + fast food).
    Budget mode  → cheapest sport + cheapest food + cinema (if non-zero).
    """
    try:
        entertainment_details = {
            "monthly_tennis_cost":     row["x40"] * 2 * 4,
            "monthly_cinema_cost":     row["x41"] * 2,
            "monthly_gym_cost":        row["x39"] * 3,
            "monthly_restaurant_cost": row["x1"]  * 4,
            "monthly_mcdo_cost":       row["x3"]  * 2,
        }

        if budget_mode:
            sports = [v for v in [entertainment_details["monthly_tennis_cost"],
                                  entertainment_details["monthly_gym_cost"]] if v > 0]
            food   = [v for v in [entertainment_details["monthly_restaurant_cost"],
                                  entertainment_details["monthly_mcdo_cost"]] if v > 0]
            cinema = entertainment_details["monthly_cinema_cost"]

            entertainment_total = (
                min(sports, default=0) +
                min(food,   default=0) +
                (cinema if cinema > 0 else 0)
            )
        else:
            entertainment_total = sum(entertainment_details.values())

        entertainment = {
            "monthly_total_cost": round(entertainment_total, 2),
            "monthly_details":    entertainment_details,
        }
        return entertainment, round(entertainment_total, 2), None

    except KeyError as e:
        return {}, 0, f"Error: Missing column {e} in compute_entertainment."
    except Exception as e:
        return {}, 0, f"Unexpected error in compute_entertainment: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 7. ACCOMMODATIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_accommodations(row, budget_mode: bool = False):
    """
    Estimate monthly rent.

    Four options (x48–x51):
        Studio city-centre · Studio outside centre
        Family apt city-centre · Family apt outside centre

    Budget mode  → cheapest available rent.
    Comfort mode → average of all available rents.
    """
    try:
        accommodations_details = {
            "monthly_studio_city_center_rent":              row["x48"],
            "monthly_studio_outside_center_rent":           row["x49"],
            "monthly_family_apartment_city_center_rent":    row["x50"],
            "monthly_family_apartment_outside_center_rent": row["x51"],
        }

        valid_rents   = [v for v in accommodations_details.values() if v > 0]
        cheapest_cost = min(valid_rents) if valid_rents else 0
        average_cost  = (sum(valid_rents) / len(valid_rents)) if valid_rents else 0

        accommodations_total = cheapest_cost if budget_mode else average_cost

        accommodations = {
            "monthly_total_cost": accommodations_total,
            "monthly_details":    accommodations_details,
        }
        return accommodations, accommodations_total, cheapest_cost, None

    except KeyError as e:
        return {}, 0, 0, f"Error: Missing column {e} in compute_accommodations."
    except Exception as e:
        return {}, 0, 0, f"Unexpected error in compute_accommodations: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY  (ML mode prediction)
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(
    salary_value,
    bills_avg,
    transport_total,
    groceries_avg,
    entertainment_total,
    accommodations_total,
    cheapest_accomodation,
    row,
):
    """
    Aggregate all expenses and determine the lifestyle mode.

    Mode prediction
    ---------------
    If the trained model (models/mode_model.pkl) is available, it receives
    the five expense totals already computed by the other functions and
    predicts "comfort" or "budget".

    Fallback (model not yet trained): budget if total_expenses > salary.

    When mode == "budget", all adjustable categories are recalculated with
    budget_mode=True so the final totals reflect realistic frugal spending.
    """
    try:
        total_expenses   = (
            bills_avg + transport_total + groceries_avg +
            entertainment_total + accommodations_total
        )
        remaining_income = salary_value - total_expenses

        # ── Mode decision ─────────────────────────────────────────────────────
        if _ML_READY:
            x = np.array([[
                bills_avg,
                transport_total,
                groceries_avg,
                entertainment_total,
                accommodations_total,
            ]])
            predicted_mode = _mode_model.predict(x)[0]
        else:
            predicted_mode = "comfort" if remaining_income > 0 else "budget"

        # ── Budget recalculation ──────────────────────────────────────────────
        entertainment_result  = None
        accommodations_result = None
        transportation_result = None
        error = ""

        if predicted_mode == "budget":
            entertainment_budget, entertainment_total_budget, ent_error = (
                compute_entertainment(row, budget_mode=True)
            )
            accommodations_budget, accommodations_total_budget, _, acc_error = (
                compute_accommodations(row, budget_mode=True)
            )
            transportation_budget, transport_total_budget, _, trans_error = (
                compute_transportation(row, budget_mode=True)
            )

            total_expenses   = (
                bills_avg + transport_total_budget + groceries_avg +
                entertainment_total_budget + accommodations_total_budget
            )
            remaining_income = salary_value - total_expenses

            entertainment_result  = entertainment_budget
            accommodations_result = accommodations_budget
            transportation_result = transportation_budget
            error = ent_error or acc_error or trans_error or ""

        summary = {
            "total_monthly_expenses": total_expenses,
            "remaining_income":       remaining_income,
        }
        return (
            summary,
            entertainment_result,
            accommodations_result,
            transportation_result,
            predicted_mode,
            error,
        )

    except Exception as e:
        return {}, None, None, None, "error", str(e)
