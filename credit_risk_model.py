import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score
)

# ============================================================
#  CREDIT RISK MODEL — Bank of America & Major US Banks
#  Finance Portfolio Project #4
# ============================================================

np.random.seed(42)

COUNTRY = "USA"

US_BANKS = [
    'Bank of America', 'JPMorgan Chase', 'Wells Fargo',
    'Citibank', 'U.S. Bancorp', 'Goldman Sachs',
    'Morgan Stanley', 'Truist Financial', 'PNC Financial', 'Capital One'
]

US_STATES = [
    'California', 'Texas', 'Florida', 'New York', 'Illinois',
    'Pennsylvania', 'Ohio', 'Georgia', 'North Carolina', 'Michigan',
    'New Jersey', 'Virginia', 'Washington', 'Arizona', 'Massachusetts'
]

# Each bank has slightly different avg loan size and risk profile
BANK_PROFILES = {
    'Bank of America':  {'income_mean': 75000, 'loan_mean': 28000, 'risk_adj':  0.00},
    'JPMorgan Chase':   {'income_mean': 85000, 'loan_mean': 32000, 'risk_adj': -0.02},
    'Wells Fargo':      {'income_mean': 68000, 'loan_mean': 24000, 'risk_adj':  0.01},
    'Citibank':         {'income_mean': 80000, 'loan_mean': 30000, 'risk_adj': -0.01},
    'U.S. Bancorp':     {'income_mean': 65000, 'loan_mean': 20000, 'risk_adj':  0.02},
    'Goldman Sachs':    {'income_mean': 120000,'loan_mean': 45000, 'risk_adj': -0.04},
    'Morgan Stanley':   {'income_mean': 115000,'loan_mean': 42000, 'risk_adj': -0.03},
    'Truist Financial': {'income_mean': 62000, 'loan_mean': 18000, 'risk_adj':  0.02},
    'PNC Financial':    {'income_mean': 67000, 'loan_mean': 22000, 'risk_adj':  0.01},
    'Capital One':      {'income_mean': 60000, 'loan_mean': 16000, 'risk_adj':  0.03},
}

# ============================================================
# 1. GENERATE SYNTHETIC LOAN DATA
# ============================================================

def generate_loan_data(n=2000):
    """
    Simulates USA loan applicant data across 10 major US banks.
    Each bank has its own income/loan/risk profile.
    Credit scores are FICO (300-850).
    """
    banks  = np.random.choice(US_BANKS,   n)
    states = np.random.choice(US_STATES,  n)

    age              = np.random.randint(21, 65, n)
    credit_score     = np.random.normal(680, 75, n).clip(300, 850)
    employment_years = np.random.randint(0, 35, n)
    num_credit_lines = np.random.randint(1, 20, n)
    missed_payments  = np.random.randint(0, 5,  n)
    loan_term        = np.random.choice([12, 24, 36, 48, 60], n)

    # Generate income and loan amount based on bank profile
    income      = np.zeros(n)
    loan_amount = np.zeros(n)
    risk_adj    = np.zeros(n)

    for i, bank in enumerate(banks):
        p = BANK_PROFILES[bank]
        income[i]      = np.random.normal(p['income_mean'], p['income_mean'] * 0.3)
        loan_amount[i] = np.random.normal(p['loan_mean'],   p['loan_mean']   * 0.3)
        risk_adj[i]    = p['risk_adj']

    income      = income.clip(20000, 500000)
    loan_amount = loan_amount.clip(2000, 100000)

    debt_to_income = (loan_amount / income).clip(0.01, 0.9)

    # Default probability
    default_prob = (
        0.4
        - 0.0005  * credit_score
        - 0.000002 * income
        + 0.5     * debt_to_income
        + 0.08    * missed_payments
        - 0.003   * employment_years
        + risk_adj
    )
    default_prob = np.clip(default_prob, 0.02, 0.98)
    default      = (np.random.rand(n) < default_prob).astype(int)

    df = pd.DataFrame({
        'Bank':             banks,
        'State':            states,
        'Age':              age,
        'Income':           income.astype(int),
        'CreditScore':      credit_score.astype(int),
        'LoanAmount':       loan_amount.astype(int),
        'LoanTerm':         loan_term,
        'DebtToIncome':     debt_to_income.round(3),
        'EmploymentYears':  employment_years,
        'NumCreditLines':   num_credit_lines,
        'MissedPayments':   missed_payments,
        'Default':          default
    })

    return df

# ============================================================
# 2. TRAIN MODELS
# ============================================================

def train_models(df):
    # Drop categorical columns for ML
    X = df.drop(['Default', 'Bank', 'State'], axis=1)
    y = df['Default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return lr, rf, scaler, X_train, X_test, X_test_s, y_train, y_test, X.columns

# ============================================================
# 3. PRINT RESULTS
# ============================================================

def print_results(lr, rf, scaler, X_test, X_test_s, y_test):
    lr_preds = lr.predict(X_test_s)
    rf_preds = rf.predict(X_test)

    print("\n" + "="*55)
    print("   CREDIT RISK MODEL — USA MAJOR BANKS")
    print("="*55)

    for name, preds in [("Logistic Regression", lr_preds), ("Random Forest", rf_preds)]:
        acc = accuracy_score(y_test, preds) * 100
        print(f"\n  {name}")
        print(f"  {'─'*40}")
        print(f"  Accuracy: {acc:.2f}%")
        print(classification_report(y_test, preds, target_names=['No Default', 'Default']))

# ============================================================
# 4. DASHBOARD
# ============================================================

def plot_dashboard(lr, rf, scaler, X_test, X_test_s, y_test, feature_names, df):

    C = {
        'bg':     '#0d1117',
        'panel':  '#161b22',
        'grid':   '#21262d',
        'text':   '#e6edf3',
        'green':  '#3fb950',
        'red':    '#f85149',
        'blue':   '#58a6ff',
        'purple': '#bc8cff',
        'yellow': '#d29922',
        'orange': '#f7931a',
    }

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(C['bg'])
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.38)

    def style(ax, title):
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['text'], labelsize=8)
        ax.set_title(title, color=C['text'], fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, color=C['grid'], linewidth=0.5, linestyle='--')
        for sp in ax.spines.values():
            sp.set_color(C['grid'])

    # ── Panel 1: ROC Curves ──
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, 'ROC Curve — Model Comparison')
    for name, model, X_in, color in [
        ("Logistic Regression", lr, X_test_s, C['blue']),
        ("Random Forest",       rf, X_test,   C['orange'])
    ]:
        probs       = model.predict_proba(X_in)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc     = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, lw=1.5, label=f'{name} (AUC={roc_auc:.3f})')
    ax1.plot([0,1],[0,1], color=C['grid'], lw=1, ls='--', label='Random (AUC=0.5)')
    ax1.set_xlabel('False Positive Rate', color=C['text'], fontsize=9)
    ax1.set_ylabel('True Positive Rate',  color=C['text'], fontsize=9)
    ax1.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 2: Feature Importance ──
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, 'Feature Importance — Random Forest')
    importances = rf.feature_importances_
    indices     = np.argsort(importances)
    bar_colors  = [C['red'] if importances[i] > np.median(importances) else C['blue']
                   for i in indices]
    ax2.barh(range(len(indices)), importances[indices], color=bar_colors, alpha=0.8)
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels([feature_names[i] for i in indices], color=C['text'], fontsize=8)
    ax2.set_xlabel('Importance', color=C['text'], fontsize=9)

    # ── Panel 3: Confusion Matrix — Logistic Regression ──
    ax3 = fig.add_subplot(gs[1, 0])
    style(ax3, 'Confusion Matrix — Logistic Regression')
    cm = confusion_matrix(y_test, lr.predict(X_test_s))
    im = ax3.imshow(cm, cmap='Blues')
    ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
    ax3.set_xticklabels(['Pred No Default', 'Pred Default'], color=C['text'], fontsize=8)
    ax3.set_yticklabels(['Actual No Default', 'Actual Default'], color=C['text'], fontsize=8)
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=14, fontweight='bold', color='white')
    plt.colorbar(im, ax=ax3)

    # ── Panel 4: Confusion Matrix — Random Forest ──
    ax4 = fig.add_subplot(gs[1, 1])
    style(ax4, 'Confusion Matrix — Random Forest')
    cm2 = confusion_matrix(y_test, rf.predict(X_test))
    im2 = ax4.imshow(cm2, cmap='Greens')
    ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
    ax4.set_xticklabels(['Pred No Default', 'Pred Default'], color=C['text'], fontsize=8)
    ax4.set_yticklabels(['Actual No Default', 'Actual Default'], color=C['text'], fontsize=8)
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm2[i,j]), ha='center', va='center',
                     fontsize=14, fontweight='bold', color='white')
    plt.colorbar(im2, ax=ax4)

    # ── Panel 5: Credit Score Distribution by Default ──
    ax5 = fig.add_subplot(gs[2, 0])
    style(ax5, 'FICO Score Distribution by Default Status')
    ax5.hist(df[df['Default']==0]['CreditScore'], bins=40,
             color=C['green'], alpha=0.6, label='No Default', density=True)
    ax5.hist(df[df['Default']==1]['CreditScore'], bins=40,
             color=C['red'],   alpha=0.6, label='Default',    density=True)
    ax5.set_xlabel('FICO Credit Score', color=C['text'], fontsize=9)
    ax5.set_ylabel('Density',           color=C['text'], fontsize=9)
    ax5.legend(fontsize=8, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 6: Default Rate by Bank ──
    ax6 = fig.add_subplot(gs[2, 1])
    style(ax6, 'Default Rate by Bank — USA')
    default_by_bank = df.groupby('Bank')['Default'].mean() * 100
    default_by_bank = default_by_bank.sort_values(ascending=True)
    bar_colors6 = [C['red'] if v > default_by_bank.mean() else C['blue']
                   for v in default_by_bank.values]
    bars = ax6.barh(default_by_bank.index, default_by_bank.values,
                    color=bar_colors6, alpha=0.8)
    for bar, val in zip(bars, default_by_bank.values):
        ax6.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', color=C['text'], fontsize=8)
    ax6.set_xlabel('Default Rate (%)', color=C['text'], fontsize=9)
    ax6.axvline(default_by_bank.mean(), color=C['yellow'], lw=1,
                ls='--', label=f'Avg {default_by_bank.mean():.1f}%')
    ax6.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])

    fig.suptitle('CREDIT RISK MODEL  —  USA MAJOR BANKS', color=C['text'],
                 fontsize=16, fontweight='bold', y=0.985)

    plt.savefig('credit_risk_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print("\nDashboard saved as credit_risk_dashboard.png")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"Generating USA loan applicant data across {len(US_BANKS)} major banks...\n")
    df = generate_loan_data(n=2000)
    print(f"Dataset    : {len(df)} applicants")
    print(f"Country    : {COUNTRY}")
    print(f"Banks      : {', '.join(US_BANKS[:5])} + {len(US_BANKS)-5} more")
    print(f"States     : {len(US_STATES)} US states")
    print(f"Default rate: {df['Default'].mean()*100:.1f}%\n")

    print("Training models...")
    lr, rf, scaler, X_train, X_test, X_test_s, y_train, y_test, feature_names = train_models(df)

    print_results(lr, rf, scaler, X_test, X_test_s, y_test)

    print("\nGenerating dashboard...")
    plot_dashboard(lr, rf, scaler, X_test, X_test_s, y_test, feature_names, df)

    print("\nDone!")