import numpy as np

def run_monte_carlo_optimization(
    initial_portfolio=1_000_000,
    initial_basis=500_000,
    ltcg_tax_rate=0.15,
    monthly_expenses=6_000,
    monthly_savings=2_000,
    
    # Probabilities
    prob_crash=0.50,            # 50% chance the "theoretical" crash actually happens
    prob_job_loss_crash=0.30,   # 30% chance of job loss if the crash happens
    prob_job_loss_normal=0.05,  # 5% baseline job loss risk if no crash
    
    # Market mechanics
    market_drop=0.40,           # 40% drop during the crash
    months_to_bottom=18,        # Takes 1.5 years to hit bottom
    months_to_recover=48,       # Takes 4 years to recover to pre-crash peak
    volatility_annual=0.15,     # 15% annual volatility
    cash_yield_annual=0.04,     # Safe cash buffer yield (e.g., HYSA)
    market_yield_annual=0.08,   # Normal market growth if no crash
    
    n_sims=500_000              # Monte Carlo paths to simulate per strategy
):
    total_months = months_to_bottom + months_to_recover
    
    # Calculate monthly rates
    sigma = volatility_annual / np.sqrt(12)
    r_cash = cash_yield_annual / 12
    r_market = market_yield_annual / 12
    
    # Required monthly drifts to hit the crash and recovery targets precisely
    mu_crash = (1 - market_drop)**(1/months_to_bottom) - 1
    mu_recov = (1 / (1 - market_drop))**(1/months_to_recover) - 1
    
    # 1. Pre-generate Universes (Crash vs. Normal market conditions)
    np.random.seed(42)
    is_crash = np.random.rand(n_sims) < prob_crash
    
    # 2. Pre-generate Job Losses & Durations
    will_lose_job = np.zeros(n_sims, dtype=bool)
    will_lose_job[is_crash] = np.random.rand(is_crash.sum()) < prob_job_loss_crash
    will_lose_job[~is_crash] = np.random.rand((~is_crash).sum()) < prob_job_loss_normal
    
    # Job loss starts randomly during the first half (the downturn)
    loss_start = np.random.randint(1, months_to_bottom + 1, size=n_sims)
    
    # Duration is random but capped so you are re-employed no later than full market recovery
    max_duration = total_months - loss_start + 1
    loss_duration = np.random.randint(1, max_duration + 1, size=n_sims)
    loss_end = loss_start + loss_duration
    
    # 3. Pre-generate Market Returns for all months
    R = np.random.normal(0, sigma, size=(n_sims, total_months))
    for t in range(total_months):
        if t < months_to_bottom:
            drift = np.where(is_crash, mu_crash, r_market)
        else:
            drift = np.where(is_crash, mu_recov, r_market)
        R[:, t] += drift
        
    # 4. Grid Search: Test selling 0% to 100% pre-emptively
    fractions_to_test = np.linspace(0, 1.0, 51)
    results = []
    
    print(f"Running {n_sims:,} parallel lifetimes per strategy...\n")
    print(f"{'Portion Sold Today':>20} | {'Expected Final Wealth':>25}")
    print("-" * 48)
    
    for f in fractions_to_test:
        # The Pre-emptive Sell & Upfront Tax Hit
        gross_sold = f * initial_portfolio
        basis_sold = f * initial_basis
        capital_gains = np.maximum(0, gross_sold - basis_sold)
        tax_paid = capital_gains * ltcg_tax_rate
        
        # Initialize parallel states
        Cash = np.full(n_sims, gross_sold - tax_paid, dtype=np.float64)
        Stocks = np.full(n_sims, (1 - f) * initial_portfolio, dtype=np.float64)
        Basis = np.full(n_sims, (1 - f) * initial_basis, dtype=np.float64)
        Market_Index = np.ones(n_sims, dtype=np.float64)
        
        # Step through time month-by-month
        for t in range(1, total_months + 1):
            Market_Index *= (1 + R[:, t-1])
            Cash *= (1 + r_cash)
            
            # Check employment status this month
            is_unemployed = will_lose_job & (t >= loss_start) & (t < loss_end)
            is_employed = ~is_unemployed
            
            # If Employed: DCA savings into stocks
            Stocks[is_employed] += monthly_savings / Market_Index[is_employed]
            Basis[is_employed] += monthly_savings
            
            # If Unemployed: Need living expenses
            need = np.zeros(n_sims, dtype=np.float64)
            need[is_unemployed] = monthly_expenses
            
            # 1st Line of Defense: The Cash Buffer
            take_from_cash = np.minimum(need, Cash)
            Cash -= take_from_cash
            need -= take_from_cash
            
            # 2nd Line of Defense: Forced Liquidation of Stocks
            mask = need > 0
            if np.any(mask):
                valid_stocks = Stocks > 0
                active = mask & valid_stocks
                
                if np.any(active):
                    s_act = Stocks[active]
                    b_act = Basis[active]
                    m_act = Market_Index[active]
                    need_act = need[active]
                    
                    # Dynamically calculate required gross liquidation to net the needed cash
                    basis_per_unit = b_act / s_act
                    gain_per_unit = np.maximum(0, m_act - basis_per_unit)
                    tax_per_unit = gain_per_unit * ltcg_tax_rate
                    net_per_unit = m_act - tax_per_unit
                    
                    units_to_sell = need_act / net_per_unit
                    
                    # Cap at bankruptcy (can't sell more than you own)
                    bankrupt = units_to_sell > s_act
                    units_to_sell[bankrupt] = s_act[bankrupt]
                    
                    s_new = s_act - units_to_sell
                    # Average cost basis updates proportionally
                    b_new = b_act * (s_new / s_act)
                    
                    Stocks[active] = s_new
                    Basis[active] = b_new
                    
        # Cycle over: Calculate True Final After-Tax Net Worth
        Portfolio_Value = Stocks * Market_Index
        Final_Gains = np.maximum(0, Portfolio_Value - Basis)
        Final_Tax = Final_Gains * ltcg_tax_rate
        
        True_Net_Worth = Cash + Portfolio_Value - Final_Tax
        Expected_Wealth = np.mean(True_Net_Worth)
        
        results.append((f, Expected_Wealth))
        
        # Print key thresholds for insight
        if round(f * 100) % 10 == 0:
            print(f"{f*100:19.0f}% | ${Expected_Wealth:24,.0f}")
            
    best_f, max_wealth = max(results, key=lambda x: x[1])
    print("-" * 48)
    print(f"Optimal portion to sell pre-emptively: {best_f*100:.0f}%")
    print(f"Expected After-Tax Net Worth:          ${max_wealth:,.0f}")
    
    return best_f

if __name__ == "__main__":
    run_monte_carlo_optimization()
