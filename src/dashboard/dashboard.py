import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing.preprocess import load_data, engineer_features
from src.scoring.risk_score import score_dataframe

BASE = Path(__file__).resolve().parents[2]
REPORTS_PATH = BASE / "reports"

sns.set_theme(style="darkgrid")
PALETTE = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


def generate_dashboard():
    print("Loading and scoring data...")
    df = load_data()
    df = engineer_features(df)
    df = score_dataframe(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("UPI Transaction Analytics & Risk Monitoring Dashboard",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Risk Distribution Pie
    ax1 = fig.add_subplot(gs[0, 0])
    risk_counts = df["risk_label"].value_counts()
    ax1.pie(risk_counts, labels=risk_counts.index,
            colors=[PALETTE[l] for l in risk_counts.index],
            autopct="%1.1f%%", startangle=140)
    ax1.set_title("Risk Distribution")

    # 2. Daily Transaction Volume
    ax2 = fig.add_subplot(gs[0, 1:])
    daily = df.groupby("date").size().reset_index(name="count")
    ax2.plot(daily["date"], daily["count"], color="#3498db", linewidth=1.5)
    ax2.fill_between(daily["date"], daily["count"], alpha=0.2, color="#3498db")
    ax2.set_title("Daily Transaction Volume")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Transactions")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    # 3. Amount Distribution by Risk
    ax3 = fig.add_subplot(gs[1, 0])
    for label, color in PALETTE.items():
        subset = df[df["risk_label"] == label]["amount"]
        ax3.hist(subset, bins=40, alpha=0.6, color=color, label=label)
    ax3.set_xlim(0, 50000)
    ax3.set_title("Amount Distribution by Risk")
    ax3.set_xlabel("Amount (INR)")
    ax3.legend()

    # 4. Hourly Transaction Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    pivot = df.pivot_table(index="day_of_week", columns="hour",
                           values="transaction_id", aggfunc="count").fillna(0)
    sns.heatmap(pivot, ax=ax4, cmap="YlOrRd", linewidths=0.3)
    ax4.set_title("Transactions by Hour & Day")
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Day (0=Mon)")

    # 5. Top Locations by High Risk
    ax5 = fig.add_subplot(gs[1, 2])
    loc_risk = df[df["risk_label"] == "High"]["location"].value_counts().head(8)
    ax5.barh(loc_risk.index, loc_risk.values, color="#e74c3c")
    ax5.set_title("High Risk Transactions by Location")
    ax5.set_xlabel("Count")

    # 6. Risk Score Distribution
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(df["risk_score"], bins=50, color="#9b59b6", edgecolor="white")
    ax6.axvline(0.35, color="#f39c12", linestyle="--", label="Medium threshold")
    ax6.axvline(0.70, color="#e74c3c", linestyle="--", label="High threshold")
    ax6.set_title("Risk Score Distribution")
    ax6.set_xlabel("Risk Score")
    ax6.legend(fontsize=8)

    # 7. Top Users by Transaction Count
    ax7 = fig.add_subplot(gs[2, 1])
    top_users = df.groupby("user_id").size().nlargest(10)
    ax7.bar(top_users.index, top_users.values, color="#1abc9c")
    ax7.set_title("Top 10 Users by Transaction Count")
    ax7.set_xlabel("User ID")
    ax7.set_ylabel("Count")
    plt.setp(ax7.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    # 8. Device Usage by Risk
    ax8 = fig.add_subplot(gs[2, 2])
    device_risk = df.groupby(["device_info", "risk_label"]).size().unstack(fill_value=0)
    device_risk.plot(kind="bar", ax=ax8,
                     color=[PALETTE[c] for c in device_risk.columns],
                     edgecolor="white")
    ax8.set_title("Device Usage by Risk Level")
    ax8.set_xlabel("Device")
    ax8.set_ylabel("Count")
    plt.setp(ax8.get_xticklabels(), rotation=30, ha="right")
    ax8.legend(title="Risk", fontsize=8)

    out_path = REPORTS_PATH / "dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    generate_dashboard()
