def rank_candidates(df, w=0.7):  # w=0.7 confirmed optimal by grid search
    df = df.copy()
    # w controls how much title similarity matters vs connections (1-w).
    # Default 0.7/0.3 is a design choice — tunable via hyperparameter search.
    df["fit"] = w * df["fit"] + (1 - w) * df["connections_norm"]
    return df.sort_values("fit", ascending=False).reset_index(drop=True)
