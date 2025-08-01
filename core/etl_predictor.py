import pandas as pd

# ✅ Calculate yield loss and value loss based on damage index
def calculate_value_loss(I, market_cost_per_kg):
    yield_lost = I * 1000
    return yield_lost, market_cost_per_kg * yield_lost

# ✅ Predict ETL days and track pest severity progression
def predict_etl_days(data):
    etl_days_list = []
    full_progress_data = []

    for row in data:
        pest_name, N_current, I_old, _, C, market_cost_per_kg, _, _, fev_con = row
        days = 0

        while days <= 28:
            if 19 <= fev_con <= 21:
                use = 1.2
            elif 22 <= fev_con <= 40:
                use = 1.4
            elif 14.25 <= fev_con <= 15.75:
                use = 0.8
            else:
                use = 1.0

            N_new = N_current * use
            I_new = (I_old / N_current) * N_new
            yield_lost_new = I_new * 1000
            value_loss_new = yield_lost_new * market_cost_per_kg
            result = value_loss_new / C if C != 0 else 0

            full_progress_data.append([
                pest_name, days, round(N_current, 3), round(I_old, 3),
                round(yield_lost_new, 3), round(value_loss_new, 3), round(result, 3)
            ])

            if result > 0.85:
                etl_days_list.append({"Pest Name": pest_name, "Days to ETL": days})
                break

            days += 7
            N_current = N_new
            I_old = I_new

    for item in etl_days_list:
        pest = item["Pest Name"]
        days = item.get("Days to ETL")
        if days is not None:
            delta = max(1, int(round(days * 0.1)))
            item["ETL Range (Days)"] = f"{max(0, days - delta)} – {days + delta} days"
        else:
            item["ETL Range (Days)"] = "Not reached within 28 days"

    df_etl = pd.DataFrame(etl_days_list)
    df_progress = pd.DataFrame(full_progress_data, columns=[
        "Pest Name", "Day", "Pest Count (N)", "Damage Index (I)",
        "Yield Loss (kg)", "Value Loss", "Value Loss / Cost"
    ])
    df_progress["Pest Severity (%)"] = (df_progress["Value Loss / Cost"] * 100).round(2)
    df_progress.drop("Value Loss / Cost", axis=1, inplace=True)

    return df_etl, df_progress
