import numpy as np
import matplotlib.pyplot as plt

def simulate_compound_poisson(lambda_val, num_months=100_000):
    """Simulates the total monthly loss for Model A and Model B."""
    
    # 1. Generate the frequency (number of claims) for all 100,000 months
    counts = np.random.poisson(lam=lambda_val, size=num_months)
    total_claims = np.sum(counts)
    
    # Pre-calculate Lognormal parameters
    mu = np.log(1000) - 0.5 * np.log(2)
    sigma = np.sqrt(np.log(2))
    
    # 2. Batch-generate every single claim for the entire simulation at once
    claims_A = np.random.exponential(scale=1000, size=total_claims)
    claims_B = np.random.lognormal(mean=mu, sigma=sigma, size=total_claims)
    
    # Arrays to store the aggregated monthly totals
    S_A = np.zeros(num_months)
    S_B = np.zeros(num_months)
    
    # 3. Sum the claims by month using a sliding index
    idx = 0
    for i in range(num_months):
        n = counts[i]
        if n > 0:
            S_A[i] = np.sum(claims_A[idx : idx+n])
            S_B[i] = np.sum(claims_B[idx : idx+n])
            idx += n
            
    return S_A, S_B

def print_metrics(lambda_val, S_A, S_B):
    """Calculates and prints the mean, variance, and tail probabilities."""
    theoretical_mean = lambda_val * 1000
    theoretical_var = lambda_val * 2_000_000  # E[X^2] = Var(X) + E[X]^2 = 1,000,000 + 1,000,000
    
    print(f"=== RESULTS FOR LAMBDA = {lambda_val} ===")
    print(f"Theoretical Mean: {theoretical_mean} | Theoretical Variance: {theoretical_var}")
    print(f"Model A (Exp)   - Mean: {np.mean(S_A):.2f} | Variance: {np.var(S_A):.2f}")
    print(f"Model B (LogN)  - Mean: {np.mean(S_B):.2f} | Variance: {np.var(S_B):.2f}")
    print("-" * 50)
    
    thresholds = [20000, 30000, 40000, 50000]
    print("Tail Probabilities P(S > u):")
    print(f"{'Threshold (u)':<15} | {'Model A (Exp)':<15} | {'Model B (LogN)':<15}")
    for u in thresholds:
        prob_A = np.sum(S_A > u) / len(S_A)
        prob_B = np.sum(S_B > u) / len(S_B)
        print(f"{u:<15} | {prob_A:<15.5f} | {prob_B:<15.5f}")
    print("\n")

if __name__ == "__main__":
    # Lock the seed for reproducible academic results
    np.random.seed(42)
    num_months = 100_000
    
    # --- PART 1, 2 & 3: Base Simulation (lambda = 20) ---
    S_A_20, S_B_20 = simulate_compound_poisson(20, num_months)
    print_metrics(20, S_A_20, S_B_20)
    
    # --- PART 4: Changing Lambda ---
    for lambd in [10, 40]:
        S_A_tmp, S_B_tmp = simulate_compound_poisson(lambd, num_months)
        print_metrics(lambd, S_A_tmp, S_B_tmp)
        
    # --- PLOTTING THE EMPIRICAL DISTRIBUTIONS ---
    plt.figure(figsize=(12, 6))
    
    # We use a logarithmic y-scale so the "tail" is visibly distinct
    plt.hist(S_A_20, bins=200, range=(0, 100_000), density=True, alpha=0.6, 
             label='Model A (Exponential - Light Tail)', color='blue')
    plt.hist(S_B_20, bins=200, range=(0, 100_000), density=True, alpha=0.6, 
             label='Model B (Lognormal - Heavy Tail)', color='red')
    
    # Mark the ruin threshold visually
    plt.axvline(x=50000, color='black', linestyle='--', label='Reserve Threshold (u=50,000)')
    
    plt.yscale('log')
    plt.title("Empirical Distribution of Total Monthly Loss (S) for λ=20\n(Logarithmic Y-Axis to Reveal Tail Behavior)")
    plt.xlabel("Total Monthly Loss (S)")
    plt.ylabel("Log Probability Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()