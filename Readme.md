# X5 Retail Hero — Discount Uplift with TensorFlow

> **Send discounts only to people who actually need them.** This project uses a neural network to predict who will buy *because* of a promotion across `200,039` retail customers.

### Why Uplift Instead of Normal Purchase Prediction?

A regular prediction model asks: *"Will this customer buy?"* That often wastes discounts on **Sure Things**—customers who would buy even without a deal.

Instead, we predict **uplift**: how much more likely a customer is to buy **if they get the offer** ($T = 1$) compared to **if they don't** ($T = 0$):

$$
\text{Uplift} = P(\text{Buy} \mid \text{Offer} = 1) - P(\text{Buy} \mid \text{Offer} = 0)
$$

| Customer Group | With Offer ($T=1$) | Without Offer ($T=0$) | What We Should Do | 
| ----- | ----- | ----- | ----- | 
| **Persuadables** | Buys | Doesn't buy | **Send the discount** | 
| **Sure Things** | Buys | Buys | Don't send (save profit margin) | 
| **Lost Causes** | Doesn't buy | Doesn't buy | Don't send (save ad spend) | 
| **Sleeping Dogs** | Doesn't buy | Buys | Don't send (avoid turning them off) | 

### How It Works

```
[Hugging Face Dataset: retailhero-uplift]
 ├── uplift_train (did they get an offer, did they buy?)
 └── clients (age, gender)
          │
          ▼  Join on client_id (200,039 customers)
[Prep Data: Scale age, encode categories, split 80% train / 20% test]
          │
          ▼  TensorFlow Neural Network (3,093 parameters)
  age (1) ───────────┐
  gender (Emb 4d) ───┼─► Combine (14) ─► Dense(64) ─► Dropout(0.2) ─► Dense(32) ─► Dropout(0.1) ─► Buy Probability
  category (Emb 8d) ─┤
  offer flag (1) ────┘
          │
          ▼  Run twice per customer: once with offer = 1, once with offer = 0
[Uplift Score = (Prob with offer) - (Prob without offer)]

```

### Results
Saving the Marketing Cost significantly by not giving unnecessary discounts and Targeting Relevent customers 

```

### Quick Start

```
pip install -q tensorflow pandas numpy scikit-learn matplotlib datasets
jupyter notebook "discount_optimiation_DL (1).ipynb"

```

### Ideas to Improve Next

* Use real favorite product categories from purchase history instead of the current placeholder value.

* Compare results against a two-model setup (T-Learner) and add uplift-specific metrics like Qini or AUUC.
