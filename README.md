# Genuity x Ethos Synthetic Data Challenge - Solution Writeup

## 🏆 Competition Overview

**Competition:** Genuity x Ethos => Synthetic Data Challenge  
**Host:** ETHOS Coding Club, IIT Guwahati & Genuity IO  
**Task:** Generate synthetic time-series data that balances accuracy, privacy, and fairness  
**Dataset Size:** 3,322 rows (60% provided, 40% hidden for evaluation)

### Objective
Build high-quality synthetic time-series datasets using Genuity's open-source library that:
- Maintains statistical similarity to real data
- Preserves privacy (no memorization)
- Achieves low detectability (realistic synthetic data)

---

## 📊 Evaluation Metrics

The final score is computed as the average of three components:

1. **Statistical Similarity** - Distribution alignment, correlations, autocorrelation structures
2. **Privacy Score** - Protection against direct leakage and memorization
3. **Detectability Score** - Difficulty in distinguishing real vs synthetic samples

**Formula:**
```
Final Score = (Statistical Similarity + Privacy Score + Detectability Score) / 3
```

---

## 🔧 My Approach

### Models Explored

I experimented with multiple synthetic data generation approaches:

| Model | Description | Performance |
|-------|-------------|-------------|
| **Genuity Model** | Competition's official library | Baseline |
| **CTGAN** | Conditional Generative Adversarial Network | ⭐ Best performing |
| **TVAE** | Tabular Variational Autoencoder | Moderate |
| **Gaussian Copulas** | Statistical copula-based approach | Moderate |
| **Copula GAN** | Hybrid copula + GAN approach | Moderate |

### Final Solution: Ensemble Approach

My final submission combined **CTGAN** with **Genuity predictions** using harmonic mean ensemble:

```python
# Ensemble strategy: Harmonic mean for numerical columns
for col in cols_to_ensemble:
    ensemble_df[col] = 2 / (1/df1[col] + 1/df2[col])
```

**Why CTGAN dominated?** After testing, CTGAN showed superior performance on the evaluation metrics, so I weighted the ensemble accordingly.

---

## 💻 Implementation Details

### Key Hyperparameters

<details>
<summary><b>Pre Processing</b></summary>

```python
# Preprocessing Done
pre = TabularPreprocessor(
    scaler_type="standard",
    encoding_strategy="onehot",
)

```
<details>
<summary><b>1. Genuity Model</b></summary>

```python
# Genuity hyperparameters
ctgan = CTGANAPI()

losses = ctgan.fit(
    data=X.values,
    continuous_cols=list(range(len(result["continuous"].columns))),
    categorical_cols=list(range(
        len(result["continuous"].columns),
        X.shape[1]
    )),
    epochs=300
)
```
</details>

<details>
<summary><b>2. CTGAN</b></summary>

```python
# CTGAN hyperparameters
ctgan = CTGANSynthesizer(
     metadata=metadata,
        epochs=100,
        generator_dim=(512, 512, 512, 512),
        discriminator_dim=(512, 512, 512, 512),
)
```
</details>

<details>
<summary><b>3. TVAE</b></summary>

```python
# TVAE hyperparameters
tvae = TVAESynthesizer(metadata, epochs=100)
```
</details>

<details>
<summary><b>4. Gaussian Copulas</b></summary>

```python
# Gaussian Copulas hyperparameters
gc = GaussianCopulaSynthesizer(
    metadata
    )
```
</details>

<details>
<summary><b>5. Copula GAN</b></summary>

```python
# Copula GAN hyperparameters
cgan = CopulaGANSynthesizer(
    metadata, epochs=200,verbose=True,
    )
```
</details>

---

## 🛠️ Submission Pipeline

### Global Submission Function

Created a robust submission function to handle all formatting requirements and avoid evaluation errors:

```python
def submit(data, subname):
    """
    Prepares and saves submission file in exact competition format
    
    Args:
        data: DataFrame with synthetic data
        subname: Name for the output CSV file
    """
    df = data.copy()
    
    # Sort by timestamp
    df = df.sort_values("t").reset_index(drop=True)
    
    # Add row_id_column_name starting from 1
    df["row_id_column_name"] = range(1, len(df) + 1)
    
    # Remove Series column if present (mandatory)
    if "Series" in df.columns:
        df = df.drop(columns=["Series"])
    
    # Exact column order required for submission
    submission_cols = [
        "row_id_column_name", "Symbol", "Prev Close", "Open", 
        "High", "Low", "Last", "Close", "VWAP", "Volume", 
        "Turnover", "Trades", "Deliverable Volume", "%Deliverble", "t"
    ]
    
    df = df[submission_cols]
    df.to_csv(f"{subname}.csv", index=False)
    
    print("Done! Matched EXACT submission format.")
    print(f"Done! Saved as {subname}.csv file submission done")
```

### Key Submission Requirements

✅ **Must have:** `row_id_column_name` as first column (0-3321)  
✅ **Exact row count:** 3,322 rows  
✅ **Remove:** 'Series' column (mandatory)  
✅ **Timestamp:** Must be in strictly ascending order  
✅ **Schema:** Must match `real_0.6.csv` exactly

---

## 🚧 Challenges Faced

### 1. Evaluation Errors
Multiple submission attempts failed due to:
- Schema mismatches
- Incorrect row counts
- Column ordering issues
- Missing/extra columns

**Solution:** Created the standardized `submit()` function to ensure consistent formatting

### 2. Model Selection
Balancing the three metrics (similarity, privacy, detectability) was challenging as models optimized for one often compromised others.

**Solution:** Ensemble approach leveraging strengths of both Genuity and CTGAN

### 3. Hyperparameter Tuning
Each model required extensive tuning to achieve optimal performance on the composite metric.

---

## 📈 Results

### Public Leaderboard Scores

| Approach | Statistical Similarity | Privacy Score | Detectability | Final Score |
|----------|----------------------|---------------|---------------|-------------|
| Genuity Baseline | [Score] | [Score] | [Score] | [Score] |
| CTGAN | [Score] | [Score] | [Score] | [Score] |
| TVAE | [Score] | [Score] | [Score] | [Score] |
| Gaussian Copulas | [Score] | [Score] | [Score] | [Score] |
| Copula GAN | [Score] | [Score] | [Score] | [Score] |
| **Final Ensemble** | **[Score]** | **[Score]** | **[Score]** | **[Score]** |

---

## 🔑 Key Takeaways

1. **CTGAN excels at time-series synthesis** - Particularly effective for financial/stock market data
2. **Ensemble methods improve robustness** - Combining models can balance competing objectives
3. **Submission format matters** - Always validate against exact competition requirements
4. **Hyperparameter tuning is crucial** - Small changes significantly impact all three metrics

---

## 📚 Resources

- **Genuity Library:** [GitHub Repository](https://github.com/S-G-mathematics/genuity_os)
- **Competition Page:** [Kaggle Competition](https://kaggle.com/competitions/genuityxethos)

---

## 🙏 Acknowledgments

- **ETHOS Coding Club, IIT Guwahati** for organizing the competition
- **Genuity IO** for providing the open-source library and mentorship
- The Kaggle community for discussions and insights

---

## 📝 Citation

```
Genuity io. Genuity x Ethos=>Synthetic Data Challenge. 
https://kaggle.com/competitions/genuityxethos, 2025. Kaggle.
```

---

**Competition Period:** September 29, 2025 - November 30, 2025  
**Prize Pool:** ₹45,000 INR + Direct hiring interviews at Genuity IO

---

## 🚀 Reproducibility

All code is available in the Kaggle Notebook. To reproduce:

1. Clone the repository or download the notebook
2. Install required libraries: `pip install genuity sdv pandas numpy`
3. Run the preprocessing pipeline
4. Train models with provided hyperparameters
5. Generate ensemble predictions
6. Use the `submit()` function to create final submission

OR

Use my Github Code which have all the Three file 
  i. Genuity Notebook
  ii. CTGAN Notebook
  iii. Ensmeble Code

---

**Author:** Ayusman Samasi 
**Contact:** hariswarsamasi@gmail.com  
**Kaggle Profile:** https://www.kaggle.com/samasiayushman/
