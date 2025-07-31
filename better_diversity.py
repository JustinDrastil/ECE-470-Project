import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === CONFIG ===
TICKERS = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU', 'AGG', 'TLT']
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
NUM_GENERATIONS = 100
POP_SIZE = 50
LAMBDA = 0.5  # risk-return tradeoff
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3
SEED = 42

np.random.seed(SEED)

# === DATA COLLECTION ===
def fetch_price_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    return df.dropna()

price_data = fetch_price_data(TICKERS, START_DATE, END_DATE)
returns = price_data.pct_change().dropna()
expected_returns = returns.mean()
cov_matrix = returns.cov()

# === SPLIT RETURNS INTO TRAIN/VAL/TEST ===
n = len(returns)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_returns = returns.iloc[:train_end]
val_returns   = returns.iloc[train_end:val_end]
test_returns  = returns.iloc[val_end:]

# Compute mean and covariance matrices
train_mu = train_returns.mean()
train_cov = train_returns.cov()

val_mu = val_returns.mean()
val_cov = val_returns.cov()

test_mu = test_returns.mean()
test_cov = test_returns.cov()


def entropy(weights):
    return -np.sum(weights * np.log(weights + 1e-8))

def fitness(weights, mu, cov, lam, entropy_weight=0.01):
    ret = np.dot(weights, mu)
    risk = np.dot(weights.T, np.dot(cov, weights))
    div_penalty = entropy(weights)
    return lam * ret - (1 - lam) * risk + entropy_weight * div_penalty

# === GA OPERATORS ===
def initialize_population(size, num_assets):
    pop = []
    for _ in range(size):
        weights = np.random.dirichlet(np.ones(num_assets))
        pop.append(weights)
    return np.array(pop)

def tournament_selection(pop, scores):
    selected = []
    for _ in range(len(pop)):
        indices = np.random.choice(len(pop), TOURNAMENT_SIZE, replace=False)
        best = indices[np.argmax(scores[indices])]
        selected.append(pop[best])
    return np.array(selected)

def crossover(parent1, parent2):
    alpha = np.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child / np.sum(child)

def mutate(weights, rate=0.1):
    if np.random.rand() < rate:
        noise = np.random.normal(0, 0.1, len(weights))
        weights += noise
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)
    return weights

def evaluate_learning_curve(returns, val_mu, val_cov, test_mu, test_cov, sizes=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
    train_scores = []
    val_scores = []
    test_scores = []
    returns_list = []
    risks = []
    sharpes = []
    weights_list = []

    for frac in sizes:
        n = int(frac * len(returns))
        train_returns = returns.iloc[:n]
        mu = train_returns.mean()
        cov = train_returns.cov()

        weights, _, _, _ = run_genetic_algorithm(mu, cov, val_mu, val_cov, test_mu, test_cov)

        # Fitness
        train_score = fitness(weights, mu, cov, LAMBDA)
        val_score = fitness(weights, val_mu, val_cov, LAMBDA)
        test_score = fitness(weights, test_mu, test_cov, LAMBDA)

        # Risk/Return/Sharpe on test set
        expected_ret = np.dot(weights, test_mu)
        variance = np.dot(weights.T, np.dot(test_cov, weights))
        std_dev = np.sqrt(variance)
        sharpe = expected_ret / std_dev if std_dev > 0 else 0

        # Store
        train_scores.append(train_score)
        val_scores.append(val_score)
        test_scores.append(test_score)
        returns_list.append(expected_ret)
        risks.append(std_dev)
        sharpes.append(sharpe)
        weights_list.append(weights)

        print(f"Training size: {n} → Train: {train_score:.5f}, Val: {val_score:.5f}, Test: {test_score:.5f}, "
              f"Return: {expected_ret:.5f}, Risk: {std_dev:.5f}, Sharpe: {sharpe:.3f}")

    return sizes, train_scores, val_scores, test_scores, returns_list, risks, sharpes, weights_list

def plot_efficient_frontier(mu, cov, lam_values=np.linspace(0, 1, 25)):
    returns = []
    risks = []

    for lam in lam_values:
        weights, _, _, _ = run_genetic_algorithm(mu, cov, mu, cov, mu, cov, lam=lam)
        ret = np.dot(weights, mu)
        risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        returns.append(ret)
        risks.append(risk)

    # Sort the (risk, return) pairs by risk for a smooth curve
    sorted_pairs = sorted(zip(risks, returns))
    sorted_risks, sorted_returns = zip(*sorted_pairs)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_risks, sorted_returns, marker='o')
    plt.title("Efficient Frontier (Train Set)")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def sensitivity_analysis(mu, cov, val_mu, val_cov, test_mu, test_cov, param='lambda'):
    values = np.linspace(0.1, 0.9, 9)
    fitness_vals = []
    returns = []
    risks = []
    sharpes = []

    for v in values:
        if param == 'lambda':
            weights, _, _, _ = run_genetic_algorithm(mu, cov, val_mu, val_cov, test_mu, test_cov, lam=v)
        elif param == 'mutation':
            global MUTATION_RATE
            old_rate = MUTATION_RATE
            MUTATION_RATE = v
            weights, _, _, _ = run_genetic_algorithm(mu, cov, val_mu, val_cov, test_mu, test_cov)
            MUTATION_RATE = old_rate
        else:
            raise ValueError("param must be 'lambda' or 'mutation'")

        ret = np.dot(weights, test_mu)
        risk = np.sqrt(np.dot(weights.T, np.dot(test_cov, weights)))
        sharpe = ret / risk if risk > 0 else 0

        fit = fitness(weights, test_mu, test_cov, LAMBDA)
        fitness_vals.append(fit)
        returns.append(ret)
        risks.append(risk)
        sharpes.append(sharpe)

        print(f"{param} = {v:.2f} → Return: {ret:.4f}, Risk: {risk:.4f}, Sharpe: {sharpe:.3f}, Fitness: {fit:.5f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(values, returns, label="Return")
    plt.plot(values, risks, label="Risk")
    plt.plot(values, sharpes, label="Sharpe Ratio")
    plt.plot(values, fitness_vals, label="Fitness")
    plt.xlabel(param.capitalize())
    plt.ylabel("Metric Value")
    plt.title(f"Sensitivity Analysis: {param.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === GA LOOP ===
def run_genetic_algorithm(mu_train, cov_train, mu_val, cov_val, mu_test, cov_test, lam=LAMBDA):
    num_assets = len(TICKERS)
    population = initialize_population(POP_SIZE, num_assets)

    best_train_fitnesses = []
    best_val_fitnesses = []
    best_test_fitnesses = []

    for gen in range(NUM_GENERATIONS):
        scores = np.array([fitness(ind, mu_train, cov_train, lam) for ind in population])
        best_index = np.argmax(scores)
        best_individual = population[best_index]

        # Record fitness for the best individual on all three sets
        best_train_fitnesses.append(fitness(best_individual, mu_train, cov_train, lam))
        best_val_fitnesses.append(fitness(best_individual, mu_val, cov_val, lam))
        best_test_fitnesses.append(fitness(best_individual, mu_test, cov_test, lam))

        # Selection and new generation
        selected = tournament_selection(population, scores)
        next_gen = []
        for i in range(0, POP_SIZE, 2):
            p1, p2 = selected[i], selected[(i+1) % POP_SIZE]
            child1 = mutate(crossover(p1, p2), MUTATION_RATE)
            child2 = mutate(crossover(p2, p1), MUTATION_RATE)
            next_gen.extend([child1, child2])
        population = np.array(next_gen)

    # Final optimal portfolio
    final_scores = np.array([fitness(ind, mu_train, cov_train, lam) for ind in population])
    best_index = np.argmax(final_scores)
    best_weights = population[best_index]

    return best_weights, best_train_fitnesses, best_val_fitnesses, best_test_fitnesses

# === RUN & PLOT ===
optimal_weights, train_fitness, val_fitness, test_fitness = run_genetic_algorithm(
    train_mu, train_cov, val_mu, val_cov, test_mu, test_cov
)

print("\nOptimal Portfolio Allocation:")
for ticker, weight in zip(TICKERS, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

# Evaluate on validation and test sets
val_score = fitness(optimal_weights, val_mu, val_cov, LAMBDA)
test_score = fitness(optimal_weights, test_mu, test_cov, LAMBDA)

print(f"\nValidation Fitness: {val_score:.4f}")
print(f"Test Fitness: {test_score:.4f}")

# === Risk/Return Breakdown ===
expected_return = np.dot(optimal_weights, test_mu)
portfolio_risk = np.dot(optimal_weights.T, np.dot(test_cov, optimal_weights))

print(f"\nRisk/Return Breakdown (on Test Set):")
print(f"Expected Return: {expected_return:.4f}")
print(f"Portfolio Risk (Variance): {portfolio_risk:.6f}")
print(f"Portfolio Std. Dev. (Volatility): {np.sqrt(portfolio_risk):.4f}")

# === PLOT FITNESS CURVES ===
plt.figure(figsize=(10, 6))
plt.plot(train_fitness, label="Train Fitness")
plt.plot(val_fitness, label="Validation Fitness")
plt.plot(test_fitness, label="Test Fitness")
plt.title("Fitness over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === LEARNING CURVE ===
sizes, train_curve, val_curve, test_curve, returns_list, risks, sharpes, weights_list = evaluate_learning_curve(
    returns.iloc[:train_end], val_mu, val_cov, test_mu, test_cov
)

# === PLOT FIXED RISK vs RETURN ===
# Sort by risk for a smooth frontier-like curve
sorted_points = sorted(zip(risks, returns_list, sizes))
sorted_risks, sorted_returns, sorted_sizes = zip(*sorted_points)

plt.figure(figsize=(8, 6))
plt.plot(sorted_risks, sorted_returns, marker='o')
for r, ret, s in zip(sorted_risks, sorted_returns, sorted_sizes):
    plt.text(r, ret, f"{int(s*100)}%", fontsize=9)

plt.title("Risk vs Return (Test Set)")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.grid(True)
plt.tight_layout()
plt.show()


# === PLOT SHARPE RATIO CURVE ===
plt.figure(figsize=(8, 6))
plt.plot([int(s * 100) for s in sizes], sharpes, marker='o')
plt.title("Sharpe Ratio vs Training Set Size")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Sharpe Ratio (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === PLOT LEARNING CURVE ===
plt.figure(figsize=(10, 6))
x = [int(s * 100) for s in sizes]
plt.plot(x, train_curve, marker='o', label="Train Fitness")
plt.plot(x, val_curve, marker='o', label="Validation Fitness")
plt.plot(x, test_curve, marker='o', label="Test Fitness")
plt.title("Learning Curve: Fitness vs Training Set Size")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

best_idx = np.argmax(val_curve)
best_weights = weights_list[best_idx]
best_size = sizes[best_idx]

print(f"\n🏆 Best Portfolio Found Using {int(best_size * 100)}% Training Data")
for ticker, w in zip(TICKERS, best_weights):
    print(f"{ticker}: {w:.4f}")

best_return = returns_list[best_idx]
best_risk = risks[best_idx]
best_sharpe = sharpes[best_idx]

print(f"Return: {best_return:.5f}, Risk: {best_risk:.5f}, Sharpe Ratio: {best_sharpe:.3f}")

plot_efficient_frontier(train_mu, train_cov)

# Sensitivity to λ
sensitivity_analysis(train_mu, train_cov, val_mu, val_cov, test_mu, test_cov, param='lambda')

# Sensitivity to mutation rate
sensitivity_analysis(train_mu, train_cov, val_mu, val_cov, test_mu, test_cov, param='mutation')


