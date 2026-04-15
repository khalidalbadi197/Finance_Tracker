import pandas as pd
from datetime import datetime
import os

CSV_FILE = "expenses.csv"

def load_expenses():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=["Date", "Category", "Amount", "Note"])

def save_expenses(df):
    df.to_csv(CSV_FILE, index=False)

def add_expense(df):
    print("\n--- Add New Expense ---")
    date = datetime.today().strftime("%Y-%m-%d")

    print("Categories: Food, Transport, Entertainment, Shopping, Bills, Other")
    category = input("Category: ").strip().capitalize()

    while True:
        try:
            amount = float(input("Amount ($): "))
            if amount <= 0:
                print("Amount must be greater than 0, try again.")
            else:
                break
        except ValueError:
            print("That's not a valid amount, please enter a number (e.g. 20 or 9.99)")

    note = input("Note (optional): ").strip()

    new_row = {"Date": date, "Category": category, "Amount": amount, "Note": note}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_expenses(df)
    print(f"Expense added: {category} - ${amount:.2f}")
    return df

def view_expenses(df):
    if df.empty:
        print("\nNo expenses yet!")
    else:
        print("\n--- Your Expenses ---")
        print(df.to_string(index=False))

def show_summary(df):
    if df.empty:
        print("No expenses yet!")
        return

    print("\n--- Summary by Category ---")
    summary = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    for category, total in summary.items():
        print(f"{category}: ${total:.2f}")

    print(f"\nTotal Spent: ${df['Amount'].sum():.2f}")
    print(f"Average Expense: ${df['Amount'].mean():.2f}")

    import matplotlib.pyplot as plt
    summary.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Spending by Category")
    plt.xlabel("Category")
    plt.ylabel("Amount ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def predict_next_month(df):
    if df.empty:
        print("No expenses yet!")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M")

    monthly = df.groupby("Month")["Amount"].sum().reset_index()
    monthly["Month_Num"] = range(1, len(monthly) + 1)

    if len(monthly) < 2:
        print("Need at least 2 months of data to predict.")
        return

    from sklearn.linear_model import LinearRegression
    import numpy as np

    X = monthly["Month_Num"].values.reshape(-1, 1)
    y = monthly["Amount"].values

    model = LinearRegression()
    model.fit(X, y)

    next_month = np.array([[len(monthly) + 1]])
    prediction = model.predict(next_month)[0]

    print("\n--- Monthly Spending ---")
    for _, row in monthly.iterrows():
        print(f"{row['Month']}: ${row['Amount']:.2f}")

    print(f"\nPredicted spending for next month: ${prediction:.2f}")

    import matplotlib.pyplot as plt
    plt.plot(monthly["Month_Num"], y, marker="o", label="Actual")
    plt.plot(len(monthly) + 1, prediction, marker="*", markersize=15, color="red", label="Prediction")
    plt.title("Monthly Spending + Prediction")
    plt.xlabel("Month")
    plt.ylabel("Amount ($)")
    plt.xticks(range(1, len(monthly) + 2))
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    df = load_expenses()

    while True:
        print("\n===== Finance Tracker =====")
        print("1. Add Expense")
        print("2. View All Expenses")
        print("3. Show Summary & Chart")
        print("4. Predict Next Month")
        print("5. Quit")
        choice = input("Choose (1-5): ").strip()

        if choice == "1":
            df = add_expense(df)
        elif choice == "2":
            view_expenses(df)
        elif choice == "3":
            show_summary(df)
        elif choice == "4":
            predict_next_month(df)
        elif choice == "5":
            print("Bye!")
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()