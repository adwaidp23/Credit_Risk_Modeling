# %% [markdown]
# # Phase 5: FICO Score Bucketing
# Dynamic Programming approach to quantize FICO scores into N optimal buckets using MSE and Log-Likelihood.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = pd.read_csv('Loan_Data.csv')

# %% [markdown]
# ## 5.1 Aggregate Data by FICO Score
# To optimize DP performance, we aggregate the 10,000 borrowers into unique FICO scores.

# %%
# Group by fico_score
fico_agg = df.groupby('fico_score').agg(
    total_count=('default', 'count'),
    defaults=('default', 'sum')
).reset_index().sort_values('fico_score')

fico_scores = fico_agg['fico_score'].values
counts = fico_agg['total_count'].values
defaults = fico_agg['defaults'].values
N_unique = len(fico_scores)

print(f"Aggregated 10,000 rows into {N_unique} unique FICO scores.")

# %% [markdown]
# ## 5.2 Precompute Cost Matrices
# We precalculate the MSE and Log-Likelihood costs for every possible contiguous range [i, j].

# %%
# Precompute cumulative sums for fast range queries
# cumulative_counts[i] = sum(counts[0...i-1])
cum_counts = np.zeros(N_unique + 1, dtype=int)
cum_defaults = np.zeros(N_unique + 1, dtype=int)

for i in range(N_unique):
    cum_counts[i+1] = cum_counts[i] + counts[i]
    cum_defaults[i+1] = cum_defaults[i] + defaults[i]

def get_range_stats(i, j):
    """Returns total borrowers and defaults in range [i, j] inclusive."""
    c = cum_counts[j+1] - cum_counts[i]
    d = cum_defaults[j+1] - cum_defaults[i]
    return c, d

# Cost matrices
cost_mse = np.zeros((N_unique, N_unique))
cost_ll = np.zeros((N_unique, N_unique))

for i in range(N_unique):
    for j in range(i, N_unique):
        c, d = get_range_stats(i, j)
        if c == 0:
            continue
        p = d / c
        # MSE: D - D^2 / C
        cost_mse[i, j] = d - (d**2 / c)
        
        # Log-Likelihood: D*log(p) + (C-D)*log(1-p)
        ll = 0
        if d > 0:
            ll += d * np.log(p)
        if c - d > 0:
            ll += (c - d) * np.log(1 - p)
        cost_ll[i, j] = ll

# %% [markdown]
# ## 5.3 Implement Dynamic Programming Bucketing Algorithms

# %%
def find_buckets_dp(n_buckets, cost_matrix, is_mse=True):
    """
    Generic DP function.
    If is_mse is True, we minimize cost.
    If is_mse is False, we maximize log-likelihood.
    """
    # Initialize DP table
    # dp[k][i] = best cost using k buckets for elements 0...i
    dp = np.zeros((n_buckets + 1, N_unique))
    pointers = np.zeros((n_buckets + 1, N_unique), dtype=int)
    
    # Base case: 1 bucket
    for i in range(N_unique):
        dp[1][i] = cost_matrix[0][i]
        
    for k in range(2, n_buckets + 1):
        for i in range(k - 1, N_unique):
            best_cost = np.inf if is_mse else -np.inf
            best_split = -1
            
            # Try all possible split points j
            for j in range(k - 2, i):
                curr_cost = dp[k-1][j] + cost_matrix[j+1][i]
                
                if is_mse:
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_split = j
                else:
                    if curr_cost > best_cost:
                        best_cost = curr_cost
                        best_split = j
                        
            dp[k][i] = best_cost
            pointers[k][i] = best_split
            
    # Backtrack to find boundaries
    boundaries = []
    curr = N_unique - 1
    for k in range(n_buckets, 1, -1):
        split = pointers[k][curr]
        boundaries.append(split)
        curr = split
        
    boundaries.reverse()
    
    # Boundaries are indices. Convert to ranges: [0, b1], [b1+1, b2], etc.
    ranges = []
    start = 0
    for b in boundaries:
        ranges.append((start, b))
        start = b + 1
    ranges.append((start, N_unique - 1))
    
    return ranges, dp[n_buckets][N_unique - 1]

def build_rating_map(ranges, method_name):
    """Generates the rating map from boundaries."""
    ratings = []
    # Highest FICO is lowest risk, so usually FICO buckets are inverted in risk.
    # We will compute default rates for each bucket and assign Ratings AAA, AA, A, etc.
    # Note: Sort by default rate descending to assign risk. Actually we just evaluate them.
    for idx, (start_idx, end_idx) in enumerate(ranges):
        c, d = get_range_stats(start_idx, end_idx)
        p = d / c if c > 0 else 0
        min_fico = fico_scores[start_idx]
        max_fico = fico_scores[end_idx]
        ratings.append({
            'Method': method_name,
            'Bucket': idx + 1,
            'Min_FICO': min_fico,
            'Max_FICO': max_fico,
            'Borrowers': c,
            'Defaults': d,
            'Default_Rate': p
        })
    df_ratings = pd.DataFrame(ratings)
    # Sort by Min_FICO to ensure it's ordered by score
    df_ratings = df_ratings.sort_values('Min_FICO')
    
    # Assign Risk Grades: 1 = Lowest Default Rate = AAA
    df_ratings = df_ratings.sort_values('Default_Rate')
    grades = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
    df_ratings['Risk_Grade'] = [grades[i] if i < len(grades) else f'G{i}' for i in range(len(df_ratings))]
    
    return df_ratings.sort_values('Min_FICO')

# %% [markdown]
# ## 5.4 Execute for N = 5, 7, 10 and Compare

# %%
results_all = []

for n_buckets in [5, 7, 10]:
    print(f"\n--- Processing for N = {n_buckets} buckets ---")
    
    ranges_mse, cost_mse_val = find_buckets_dp(n_buckets, cost_mse, is_mse=True)
    df_mse = build_rating_map(ranges_mse, 'MSE')
    
    ranges_ll, cost_ll_val = find_buckets_dp(n_buckets, cost_ll, is_mse=False)
    df_ll = build_rating_map(ranges_ll, 'Log-Likelihood')
    
    # Save maps
    df_mse.to_csv(f'FICO_Rating_Map_MSE_{n_buckets}.csv', index=False)
    df_ll.to_csv(f'FICO_Rating_Map_LL_{n_buckets}.csv', index=False)
    
    results_all.append(df_mse)
    results_all.append(df_ll)

    # Plot Comparison
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    
    sns.barplot(x='Bucket', y='Default_Rate', data=df_mse, ax=ax[0], palette='Blues_d')
    ax[0].set_title(f'MSE Method (N={n_buckets}) - Default Rate by Bucket')
    for i, row in df_mse.iterrows():
        ax[0].text(row['Bucket']-1, row['Default_Rate']+0.02, f"{row['Min_FICO']}-{row['Max_FICO']}\n{row['Risk_Grade']}", ha='center', fontsize=9)
        
    sns.barplot(x='Bucket', y='Default_Rate', data=df_ll, ax=ax[1], palette='Oranges_d')
    ax[1].set_title(f'Log-Likelihood Method (N={n_buckets}) - Default Rate by Bucket')
    for i, row in df_ll.iterrows():
        ax[1].text(row['Bucket']-1, row['Default_Rate']+0.02, f"{row['Min_FICO']}-{row['Max_FICO']}\n{row['Risk_Grade']}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'Buckets_Comparison_N{n_buckets}.png')
    plt.close()

print("\nAll bucketing maps generated and saved.")

# %% [markdown]
# ## 5.5 Summarize Sensitivity
# Reviewing the results across different N.

# %%
df_all = pd.concat(results_all)
print("\nSnapshot of N=5 Log-Likelihood Rating Map:")
print(df_all[(df_all['Method'] == 'Log-Likelihood') & (len(df_all['Risk_Grade']) > 0)].head(10))

# Save a sensitivity overview
df_all.to_csv('FICO_Bucketing_Sensitivity.csv', index=False)
print("Phase 5 Completed Successfully.")
