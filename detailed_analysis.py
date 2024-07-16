import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import numpy as np


months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'aug', 'sep', 'oct', 'nov', 'dec']


def read_csv(month):
    folder = f'listings_{month}'
    file_path = os.path.join(folder, 'listings.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['month'] = month  # Add a column to identify the month
        return df
    else:
        print(f"File {file_path} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if the file does not exist


def read_all_csvs(months):
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(read_csv, months))
    return pd.concat(dfs, ignore_index=True)


# Read all CSV files and combine them into a single DataFrame
combined_df = read_all_csvs(months)

# Filter listings by Queenstown district ID
qt_district_id = 70
qt_district_listings = combined_df.loc[combined_df['region_parent_id'] == qt_district_id]

# Filter listings with reviews per month >= 1
qt_listings_w_const_reviews = qt_district_listings.loc[qt_district_listings['reviews_per_month'] >= 1]

# Define filter criteria for property types, bathrooms, bedrooms, and beds
property_types = ['Entire home', 'Entire vacation home', 'Entire apartment', 'Entire place',
                  'Entire home/apt', 'Entire serviced apartment', 'Entire townhouse', 'Entire rental unit']
bathrooms = [2, 2.5]
bedrooms = [3]
beds = [3, 4]

# Apply filters using loc
filtered_listings = qt_listings_w_const_reviews.loc[
    (qt_listings_w_const_reviews['property_type'].isin(property_types)) &
    ((qt_listings_w_const_reviews['bathrooms'].isin(bathrooms) | qt_listings_w_const_reviews['bathrooms'].isnull())) &
    ((qt_listings_w_const_reviews['bedrooms'].isin(bedrooms)) |
     (qt_listings_w_const_reviews['bedrooms'].isnull() & qt_listings_w_const_reviews['beds'].isin(beds)))
]

# Remove listings where price is not available
filtered_listings = filtered_listings.dropna(subset=['price'])

# Remove dollar signs and commas from the price column and convert to float
filtered_listings.loc[:, 'price'] = filtered_listings['price'].str.replace('$', '', regex=False)\
    .str.replace(',', '', regex=False).astype(float)


# Define simulation parameters
n_simulations = 10000
occupancy_rates = [0.755, 0.71, 0.93, 0.36, 0.76, 0.84, 0.36, 0.4, 0.5, 0.6]
# Fit gde model to data
kernel = gaussian_kde(occupancy_rates)
x = np.linspace(0, 1, 1000)
pdf = kernel(x)


# Simulate occupancy rates
def bounded_resample(kernel, size, lower_bound=0, upper_bound=1):
    samples = []
    while len(samples) < size:
        sample = kernel.resample(1)[0][0]
        if lower_bound <= sample <= upper_bound:
            samples.append(sample)
    return np.array(samples)


# Generate 1000 random selections bounded between 0 and 1
sim_occupancy_rates = bounded_resample(kernel, n_simulations)

# Stay durations
stay_length_stats = [1022, 1520, 1264, 353, 244, 47, 82]
stay_length_nights_prob = stay_length_stats / np.sum(stay_length_stats)
sim_stay_lengths = []
n_nights = sim_occupancy_rates * 365
for n_night_this_sim in n_nights:
    current_nights = 0
    stay_lengths = []
    while current_nights < n_night_this_sim:
        stay_length = np.random.choice([1, 2, 3, 4, 5, 6, 7], p=stay_length_nights_prob)
        if current_nights + stay_length <= n_night_this_sim:
            stay_lengths.append(stay_length)
            current_nights += stay_length
        else:
            break
    sim_stay_lengths.append(np.mean(stay_lengths))
sim_stay_lengths = np.nan_to_num(sim_stay_lengths, nan=1)
sim_stay_lengths = np.array(sim_stay_lengths)


kernels = {}
months = filtered_listings['month'].unique()
for month in months:
    monthly_df = filtered_listings[filtered_listings['month'] == month]
    if not monthly_df.empty:
        costs = monthly_df['price'].values
        kernels[month] = gaussian_kde(costs)


def simulate_monthly_cost(df, month):
    monthly_df = df[df['month'] == month]
    if not monthly_df.empty:
        simulated_costs = bounded_resample(kernels[month], n_simulations, lower_bound=0, upper_bound=10000)
        return simulated_costs
    else:
        return np.array([])


months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sim_night_cost_by_month = []
for month in months:
    if month != 'jul':
        simulated_costs = simulate_monthly_cost(filtered_listings, month)
        sim_night_cost_by_month.append(simulated_costs)
    else:
        june_costs = simulate_monthly_cost(filtered_listings, 'jun')
        aug_costs = simulate_monthly_cost(filtered_listings, 'aug')
        july_costs = (june_costs + aug_costs) / 2
        sim_night_cost_by_month.append(july_costs)
# Calculate the average cost per night for the year
sim_night_cost_annual_avgs = np.mean(sim_night_cost_by_month, axis=0)

# Simulate average cleaning fee as triangular between 40 and 60, with mode at 55
sim_cleaning_fees_ph = np.random.triangular(left=40, mode=55, right=60, size=n_simulations)

# Simulate Airbnb fee as triangular between 3% and 3.5% with mode at 3.25%
sim_airbnb_fees = np.random.triangular(left=0.03, mode=0.0325, right=0.035, size=n_simulations)

# Simulate management fees as triangular between 10% and 25% with mode at 15%
sim_management_fees = np.random.triangular(left=0.1, mode=0.15, right=0.25, size=n_simulations)

# Simulate inspection fees between 50 and 150 (uniform)
sim_inspection_fees = np.random.uniform(low=50, high=150, size=n_simulations)

# Simulate booking fees between 0 and 50 (uniform)
sim_booking_fees = np.random.uniform(low=0, high=50, size=n_simulations)

# Simulate marketing fees between 300 and 700 (uniform)
sim_marketing_fees = np.random.uniform(low=300, high=800, size=n_simulations)

# Simulated revenue
sim_revenue = sim_occupancy_rates * sim_night_cost_annual_avgs * 365

# Sim cleaning cost 
sim_cleaning_cost = sim_cleaning_fees_ph * n_nights / sim_stay_lengths

# Sim Airbnb cost
sim_airbnb_cost = sim_airbnb_fees * sim_revenue

# Sim management cost
sim_management_cost = sim_management_fees * sim_revenue

# Sim booking cost
sim_booking_cost = sim_booking_fees * n_nights / sim_stay_lengths

# Sim total cost
sim_total_cost = sim_cleaning_cost + sim_airbnb_cost + sim_management_cost + sim_inspection_fees + sim_booking_cost\
    + sim_marketing_fees

# Sim gross profit
sim_gross_profit = sim_revenue - sim_total_cost

# Mortgage
mortgage = 5992.35 * 12
insurance = 141.75 * 12
rates = 4595.20 * (1 + np.where((sim_occupancy_rates * 365) > 90,
                                np.random.triangular(left=0.15, mode=0.2, right=0.25, size=n_simulations),
                                np.where((sim_occupancy_rates * 365) > 180,
                                         np.random.triangular(left=0.5, mode=0.65, right=0.8, size=n_simulations),
                                         0)))
maintenance = np.random.normal(loc=500, scale=250, size=n_simulations)
maintenance = np.clip(maintenance, 0, 100000)

# Sim net profit
sim_net_profit = sim_gross_profit - mortgage - insurance - rates - maintenance

# Calculate the 5th and 95th percentiles for net profit
percentile_5 = np.percentile(sim_net_profit, 5)
percentile_95 = np.percentile(sim_net_profit, 95)

# Calculate the probability of making a loss
prob_loss = np.mean(sim_net_profit < 0)

# Calculate the probability that the net profit is less than 10400
prob_less_than_10400 = np.mean(sim_net_profit < -10400)

# Calculate the weekly cost where 50% of the density lies below this value
weekly_cost_50th_percentile = -np.percentile(sim_net_profit, 50) / 52

# Calculate the 10th percentile as a weekly cost
weekly_cost_10th_percentile = -np.percentile(sim_net_profit, 10) / 52

# Print the results
print(f'5th percentile net profit: {percentile_5:.2f}')
print(f'95th percentile net profit: {percentile_95:.2f}')
print(f'Probability of making a loss: {prob_loss:.2f}')
print((f'Probability of net profit less than -$10,400 '
      f'(i.e., paying more than $200 extra per week on the house): {prob_less_than_10400*100:.2f}%'))
print((f'Expected weekly cost (i.e., 50:50 that the cost will be higher or lower than this value): '
       f'{weekly_cost_50th_percentile:.2f}'))
print((f'Weekly cost at the 10th percentile (i.e., 90% chance that the weekly cost will'
       f' be less than this): {weekly_cost_10th_percentile:.2f}'))

# Plot the net profit distribution
plt.figure(figsize=(12, 6))
plt.hist(sim_net_profit, bins=50, density=True, alpha=0.6)
plt.axvline(percentile_5, color='r', linestyle='--', label='5th percentile')
plt.axvline(percentile_95, color='b', linestyle='--', label='95th percentile')
plt.axvline(0, color='g', linestyle='--', label='Break-even')
plt.axvline(-10400, color='m', linestyle='--', label='Extra $200 per week')
plt.axvline(-weekly_cost_50th_percentile * 52, color='k', linestyle='--', label='50th percentile weekly cost')
plt.xlabel('Net Profit')
plt.ylabel('Density')
plt.title('Net Profit Distribution')
plt.legend()
plt.savefig('net_profit_distribution.png')
