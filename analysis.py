import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


listings = pd.read_csv('listings.csv')
qt_district_id = 70
qt_district_listings = listings[listings['region_parent_id'] == qt_district_id]
reviews = pd.read_csv('reviews.csv')
review_count_by_listing = reviews['listing_id'].value_counts()
qt_district_listings['review_count'] = qt_district_listings['id'].map(review_count_by_listing)
qt_district_listings_with_reviews = qt_district_listings[qt_district_listings['review_count'] > 5]
qt_listings_w_const_reviews = qt_district_listings_with_reviews[qt_district_listings_with_reviews['reviews_per_month'] >= 1]
property_types = ['Entire home',
                  'Entire vacation home',
                  'Entire apartment',
                  'Entire place',
                  'Entire home/apt',
                  'Entire serviced apartment',
                  'Entire townhouse',
                  'Entire rental unit']
bathrooms = [2, 2.5]
bedrooms = [3]
beds = [3]
# I could filter on amenities too.

filtered_listings = qt_district_listings_with_reviews[
    (qt_district_listings_with_reviews['property_type'].isin(property_types)) &
    (qt_district_listings_with_reviews['bathrooms'].isin(bathrooms)) &
    (qt_district_listings_with_reviews['bedrooms'].isin(bedrooms)) &
    (qt_district_listings_with_reviews['beds'].isin(beds))
]
filtered_listings['price'] = filtered_listings['price'].str.replace('$', '').str.replace(',', '').astype(float)
# Plot a histogram of the price
plt.figure(figsize=(12, 6))
sns.histplot(data=filtered_listings, x='price', bins=20)
plt.title('Price distribution of listings')
plt.show()


ids_of_interest = filtered_listings['id']

filtered_reviews = reviews[reviews['listing_id'].isin(ids_of_interest)]
filtered_reviews['date'] = pd.to_datetime(filtered_reviews['date'])
filtered_reviews['month'] = filtered_reviews['date'].dt.month
reviews_by_month = filtered_reviews.groupby('month').size().reset_index(name='count')
reviews_by_month['month'] = reviews_by_month['month'].replace({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                                                               6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                                                               11: 'Nov', 12: 'Dec'})

plt.figure(figsize=(12, 6))
sns.barplot(data=reviews_by_month, x='month', y='count')
plt.title('Number of reviews by month')
plt.show()

reviews_by_listing = filtered_reviews.groupby(['listing_id', 'month']).size().reset_index(name='count')
avg_reviews_by_month = reviews_by_listing.groupby('month')['count'].mean().reset_index(name='avg_count')
avg_reviews_per_month = reviews_by_listing['count'].mean()

pct_that_leaves_review = 0.8
estimate_avg_reviews_per_month = avg_reviews_per_month / pct_that_leaves_review

estimate_monthly_income = filtered_listings['price'].median() * estimate_avg_reviews_per_month
print(f'Estimated monthly income: ${estimate_monthly_income:.2f}')
annual_income = estimate_monthly_income * 12
