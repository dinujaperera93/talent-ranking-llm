def rank_candidates(df):
    df = df.copy()
    df["fit"] = 0.7 * df["fit"] + 0.3 * df["connections_norm"]
    return df.sort_values("fit", ascending=False).reset_index(drop=True)
