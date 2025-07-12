import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
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
    df = yf.download(tickers, start=start, end=end)['Adj Close']
    return df.dropna()

price_data = fetch_price_data(TICKERS, START_DATE, END_DATE)
returns = price_data.pct_change().dropna()
expected_returns = returns.mean()
cov_matrix = returns.cov()

# === FITNESS FUNCTION ===
def fitness(weights, mu, cov, lam):
    ret = np.dot(weights, mu)
    risk = np.dot(weights.T, np.dot(cov, weights))
    return lam * ret - (1 - lam) * risk

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

# === GA LOOP ===
def run_genetic_algorithm():
    num_assets = len(TICKERS)
    population = initialize_population(POP_SIZE, num_assets)
    best_fitnesses = []

    for gen in range(NUM_GENERATIONS):
        scores = np.array([fitness(ind, expected_returns, cov_matrix, LAMBDA) for ind in population])
        best_fitnesses.append(np.max(scores))

        # Selection
        selected = tournament_selection(population, scores)

        # Crossover & Mutation
        next_gen = []
        for i in range(0, POP_SIZE, 2):
            p1, p2 = selected[i], selected[(i+1) % POP_SIZE]
            child1 = mutate(crossover(p1, p2), MUTATION_RATE)
            child2 = mutate(crossover(p2, p1), MUTATION_RATE)
            next_gen.extend([child1, child2])
        population = np.array(next_gen)

    # Final best solution
    final_scores = np.array([fitness(ind, expected_returns, cov_matrix, LAMBDA) for ind in population])
    best_index = np.argmax(final_scores)
    best_weights = population[best_index]
    return best_weights, best_fitnesses

# === RUN & PLOT ===
optimal_weights, fitness_curve = run_genetic_algorithm()

print("\nOptimal Portfolio Allocation:")
for ticker, weight in zip(TICKERS, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

plt.plot(fitness_curve)
plt.title("Fitness over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()
