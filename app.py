import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.linear_model import LinearRegression
import numpy as np

CSV_FILE = "expenses.csv"

def load_expenses():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=["Date", "Category", "Amount", "Note"])

def save_expenses(df):
    df.to_csv(CSV_FILE, index=False)

# --- Page Setup ---
st.title("Personal Finance Tracker")
st.write("Track your expenses, visualize your spending, and predict next month's budget.")

df = load_expenses()

# --- Add Expense ---
st.header("Add Expense")
category = st.selectbox("Category", ["Food", "Transport", "Entertainment", "Shopping", "Bills", "Other"])
amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)
note = st.text_input("Note (optional)")

if st.button("Add Expense"):
    new_row = {"Date": datetime.today().strftime("%Y-%m-%d"), "Category": category, "Amount": amount, "Note": note}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_expenses(df)
    st.success(f"Added: {category} - ${amount:.2f}")
    st.rerun()

# --- View Expenses ---
st.header("Your Expenses")
if df.empty:
    st.write("No expenses yet.")
else:
    st.dataframe(df)

    # --- Edit Expense ---
    st.subheader("Edit an Expense")
    edit_index = st.number_input("Row number to edit (starting from 0)", min_value=0, max_value=len(df)-1, step=1)
    edit_row = df.iloc[int(edit_index)]

    new_category = st.selectbox("New Category", ["Food", "Transport", "Entertainment", "Shopping", "Bills", "Other"], index=["Food", "Transport", "Entertainment", "Shopping", "Bills", "Other"].index(edit_row["Category"]) if edit_row["Category"] in ["Food", "Transport", "Entertainment", "Shopping", "Bills", "Other"] else 0)
    new_amount = st.number_input("New Amount ($)", min_value=0.01, step=0.01, value=float(edit_row["Amount"]))
    new_note = st.text_input("New Note", value=str(edit_row["Note"]))

    if st.button("Save Edit"):
        df.at[int(edit_index), "Category"] = new_category
        df.at[int(edit_index), "Amount"] = new_amount
        df.at[int(edit_index), "Note"] = new_note
        save_expenses(df)
        st.success("Expense updated!")
        st.rerun()

    # --- Delete Expense ---
    st.subheader("Delete an Expense")
    delete_index = st.number_input("Row number to delete (starting from 0)", min_value=0, max_value=len(df)-1, step=1, key="delete")

    if st.button("Delete Expense"):
        df = df.drop(index=int(delete_index)).reset_index(drop=True)
        save_expenses(df)
        st.success("Expense deleted!")
        st.rerun()

# --- Summary Chart ---
st.header("Spending by Category")
if not df.empty:
    if st.toggle("Show Chart"):
        summary = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        ax.bar(summary.index, summary.values, color="steelblue", edgecolor="black")
        ax.set_xlabel("Category")
        ax.set_ylabel("Amount ($)")
        ax.set_title("Spending by Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# --- Prediction ---
st.header("Predict Next Month")
if not df.empty:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Amount"].sum().reset_index()
    monthly["Month_Num"] = range(1, len(monthly) + 1)

    if len(monthly) < 2:
        st.write("Need at least 2 months of data to predict.")
    else:
        if st.toggle("Show Prediction Chart"):
            X = monthly["Month_Num"].values.reshape(-1, 1)
            y = monthly["Amount"].values
            model = LinearRegression()
            model.fit(X, y)
            prediction = model.predict([[len(monthly) + 1]])[0]

            st.write(f"Predicted spending for next month: **${prediction:.2f}**")

            fig2, ax2 = plt.subplots()
            ax2.plot(monthly["Month_Num"], y, marker="o", label="Actual")
            ax2.plot(len(monthly) + 1, prediction, marker="*", markersize=15, color="red", label="Prediction")
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Amount ($)")
            ax2.set_title("Monthly Spending + Prediction")
            ax2.set_xticks(range(1, len(monthly) + 2))
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)