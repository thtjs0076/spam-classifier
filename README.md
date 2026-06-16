# Email Spam Classification — Baseline to Transformer

I built three spam classifiers of increasing complexity on the Enron email dataset, then asked a harder question than "which scores highest": **is the extra complexity actually worth it?**

## What I Found

| Model                              | Precision | F1    | Inference / Email |
| ---------------------------------- | --------- | ----- | ----------------- |
| Logistic Regression (bag-of-words) | 0.981     | 0.986 | ~1 µs (CPU)       |
| Logistic Regression (TF-IDF)       | 0.975     | 0.985 | ~0.4 µs (CPU)     |
| DistilBERT (fine-tuned)            | 0.992     | 0.993 | ~9 ms (GPU)       |

*Test set, mean over multiple seeds (LR: 5, DistilBERT: 3). Full metrics are available in the notebooks.*

EDA revealed that spam emails were characterized by a small set of highly predictive promotional terms, while ham emails contained Enron-specific organizational language. This strong lexical separation helps explain why simple linear models perform nearly as well as transformers on this dataset.

## Three Takeaways I'd Actually Defend in a Review

1. **Precision is the metric that matters here, not F1.** A false positive sends a legitimate email to the spam folder, which is the more costly mistake. Ranking by precision changes the model ordering: DistilBERT performs best, while TF-IDF, despite achieving the highest recall, falls to last place. F1 alone would have hidden this trade-off.

2. **DistilBERT's advantage appears genuine; TF-IDF's does not.** Across multiple random seeds, DistilBERT consistently outperforms the baseline by several standard deviations, whereas the Logistic Regression and TF-IDF models overlap within their experimental variance and are effectively tied.

3. **The simple model is the pragmatic choice.** DistilBERT delivers the strongest predictive performance but requires substantially greater computational resources and inference latency. The dataset's strong word-level signal allows Logistic Regression to capture most of the available information at a fraction of the cost. The preferred model therefore depends on the relative importance of predictive performance versus computational efficiency.

## What Surprised Me

I expected DistilBERT to significantly outperform the linear baselines. Instead, the performance gap was surprisingly small, suggesting that the classification task is driven largely by easily separable lexical features rather than deeper contextual understanding.

This highlights an important lesson: more sophisticated models do not necessarily produce meaningful gains when the underlying signal is already captured effectively by simple features.

## One Thing I'm Glad I Checked

The initial scores were close to 0.99—high enough that I did not fully trust them. I investigated potential data leakage and discovered that approximately 9.6% of the dataset consisted of duplicate emails.

After removing duplicates and repeating the experiments, performance remained largely unchanged. This increased confidence that the results reflected genuine signal in the data rather than an artifact of leakage. The full analysis is included in `03_transformer.ipynb`.

## Error Analysis

I manually reviewed the baseline's mistakes on the test set: 40 false positives (ham flagged as spam) versus only 16 false negatives. The model errs toward over-flagging — which, for spam, is the more harmful direction, reinforcing why precision matters here.

The false positives were mostly short or context-light internal emails ("how bout this one", brief rate-change notes) where there simply wasn't enough text for the model to read them as legitimate. The false negatives were the more interesting case: stock-promotion and marketing spam ("stocks in an uptrend", "oil and gas ... about to explode") that slipped through because their vocabulary overlaps with genuine Enron business mail — an energy company naturally discusses stocks, oil, and gas. So the hardest cases are where lexical cues point the wrong way.

## Project Structure

```text
notebooks/
00_data_check.ipynb   data loading + first look
01_eda.ipynb          what separates spam from ham
02_models.ipynb       baseline + TF-IDF (local)
03_transformer.ipynb  DistilBERT + leakage check (Colab GPU)
```

## Dataset & Tools

* Dataset: Enron-Spam (33,716 emails → 30,494 after deduplication, ~51% spam)
* Stack: Python, pandas, scikit-learn, Hugging Face Transformers, PyTorch

Run:

```bash
pip install -r requirements.txt
```

Then execute the notebooks in order (`00 → 03`).

## If I Took This Further

Many of the ham signals are specific to Enron (employee names, internal systems, and company-specific terminology), which raises questions about generalization. Future work would focus on evaluating performance under distribution shift by testing the models on a different email source rather than pursuing marginal benchmark improvements.

A secondary extension would be packaging the final model into a lightweight Streamlit application for interactive predictions.
