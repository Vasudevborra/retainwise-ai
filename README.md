# LLM-Powered Customer Intelligence Studio: Behavioral Segmentation & Decision Engine

An end-to-end Customer Intelligence and Behavioral Segmentation prototype designed for MBA-level marketing analytics and customer strategy.

The core philosophy of this system:
> **"Move from WHO the customer is demographically to HOW the customer actually behaves."**

---

## 1. System Architecture

The solution uses a **two-engine decoupled architecture**:

```
[Raw Event Streams: Transactions, Web/App Interactions, Customer Records]
                                │
                                ▼
         [Feature Engineering & Robust Normalization Engine]
       (RFM, Discount Dependency Index, Category Shannon Entropy,
        Cart Abandonment Rate, Engagement Decay Velocity)
                                │
                                ▼
             [Unsupervised ML Clustering Engine]
       (K-Means / GMM with Silhouette Score Optimization: K=5)
                                │
                                ▼
      [Cognitive LLM Customer Intelligence Layer (Gemini)]
       • Grounded In-Context Profiling (16 Strategic Dimensions)
       • Epistemic Guardrails: Observed Behavior vs. Marketing Hypothesis
       • Evidence-Grounded Natural Language Query (NLQ) Assistant
       • Parametric Campaign Brief Generator
                                │
                                ▼
           [Customer Intelligence Interactive Dashboard]
       (Executive KPIs, 2D Behavioral Map, Segment Explorer,
        Customer 360 Explainability, Margin Defense Simulator)
```

---

## 2. Quickstart & Installation

### Prerequisites
- Python 3.10+
- Required packages (pre-installed): `pandas`, `numpy`, `scikit-learn`, `flask`, `requests`

### Running the Application

In your terminal or PowerShell:
```powershell
cd C:\Users\Personal\.gemini\antigravity\scratch\customer-intelligence-dashboard
python run.py
```

The orchestrator will automatically:
1. Verify/generate realistic datasets in `data/` (`customers.csv`, `transactions.csv`, `interactions.csv`).
2. Run the Feature Pipeline to build `data/customer_features.csv`.
3. Train the K-Means model to discover behavioral segments and export `data/segment_centroids.json`.
4. Launch the local web server at `http://127.0.0.1:5000`.

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 3. Discovered Behavioral Segments

| Segment | Empirical Profile | Strategic Marketing Posture |
| :--- | :--- | :--- |
| **High-Velocity Brand Champions** | Frequent (14.8x/yr), high AOV ($164), 92% full-price, mobile app native. | **Withhold discounts to protect margin**. Reward with VIP early access drops & personal concierge reservations. |
| **Deal-Driven Opportunity Hunters** | Extreme price sensitivity (84% promo dependency), high cart abandonment on full-price items. | Enforce spend hurdles (e.g. "$25 off $120+") or multi-item bundles ("Buy 2, Get 1 Free") to guarantee positive unit contribution. |
| **High-Value Dormant Elites** | Top-decile historical lifetime spend ($1,200+), but zero transactions in 120+ days. | **Immediate white-glove re-engagement**. Send complimentary luxury gift ($0 discount) with personalized note. |
| **Digital Window Shoppers** | High web sessions (12+/mo), 88% cart abandonment occurring at shipping screen, low conversion (<1%). | Provide first-order free shipping thresholds and customer testimonial highlights to alleviate purchase hesitation. |
| **Steady Low-Maintenance Essentials** | Clockwork repeat purchases on staple SKUs, low service tickets, low content engagement. | Shift to automated 1-click subscription replenishment with 5% recurring restock perks. |

---

## 4. API Reference

- `GET /api/health`: Health status check.
- `GET /api/segments`: Returns calculated clusters, Silhouette evaluation score, and 16-dimension LLM strategy profiles.
- `GET /api/customers?limit=50`: Returns list of customer records with segment assignments.
- `GET /api/customer/<customer_id>`: Returns single customer 360 profile with feature deviations relative to global median.
- `POST /api/nlq`: Accepts `{"question": "..."}` and returns evidence-grounded response.
- `POST /api/campaign/generate`: Accepts campaign parameters and generates production copy deck.
- `POST /api/pipeline/run`: Re-executes the entire ML pipeline on demand.
