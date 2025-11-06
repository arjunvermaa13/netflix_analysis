import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!\n")

# =============================================
# STEP 2: LOAD NETFLIX DATASET
# =============================================

print("📥 Loading Netflix dataset...")
print("Choose one of these methods:\n")

# METHOD 1: Upload from your computer
print("METHOD 1: Upload CSV file")
print("Run this cell and upload the file when prompted:\n")

from google.colab import files
uploaded = files.upload()

# Get the filename
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# METHOD 2: Load directly from Kaggle (if you have the direct link)
# Uncomment these lines if you want to use Kaggle API
# !pip install kaggle -q
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d shivamb/netflix-shows
# !unzip netflix-shows.zip
# df = pd.read_csv('netflix_titles.csv')

print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# =============================================
# STEP 3: DATA EXPLORATION
# =============================================

print("=" * 60)
print("📊 DATASET OVERVIEW")
print("=" * 60)

# Display first few rows
print("\n🔍 First 5 rows:")
print(df.head())

print("\n📋 Dataset Info:")
print(df.info())

print("\n📈 Dataset Statistics:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

print("\n🏷️ Column Names:")
print(df.columns.tolist())

# =============================================
# STEP 4: DATA CLEANING
# =============================================

print("\n" + "=" * 60)
print("🧹 DATA CLEANING")
print("=" * 60)

# Make a copy
netflix_df = df.copy()

# Fill missing values
netflix_df['director'].fillna('Unknown', inplace=True)
netflix_df['cast'].fillna('Unknown', inplace=True)
netflix_df['country'].fillna('Unknown', inplace=True)
netflix_df['date_added'].fillna('Unknown', inplace=True)
netflix_df['rating'].fillna('Unknown', inplace=True)

# Convert date_added to datetime
netflix_df['date_added'] = pd.to_datetime(netflix_df['date_added'], errors='coerce')

# Extract year, month from date_added
netflix_df['year_added'] = netflix_df['date_added'].dt.year
netflix_df['month_added'] = netflix_df['date_added'].dt.month
netflix_df['month_name'] = netflix_df['date_added'].dt.month_name()

# Extract duration numbers
netflix_df['duration_int'] = netflix_df['duration'].str.extract('(\d+)').astype(float)

print("✅ Data cleaning completed!")
print(f"\n📊 Cleaned dataset shape: {netflix_df.shape}")

# =============================================
# STEP 5: SQL-STYLE ANALYSIS USING PANDAS
# =============================================

print("\n" + "=" * 60)
print("🔍 SQL-STYLE ANALYSIS (Using Pandas)")
print("=" * 60)

# QUERY 1: Content Type Distribution
print("\n1️⃣ CONTENT TYPE DISTRIBUTION")
print("-" * 40)
content_type = netflix_df['type'].value_counts()
content_type_pct = (content_type / len(netflix_df) * 100).round(2)
content_distribution = pd.DataFrame({
    'Count': content_type,
    'Percentage': content_type_pct
})
print(content_distribution)

# QUERY 2: Top 10 Countries
print("\n2️⃣ TOP 10 COUNTRIES BY CONTENT")
print("-" * 40)
# Split countries and count
countries = netflix_df['country'].str.split(',', expand=True).stack()
countries = countries.str.strip()
top_countries = countries.value_counts().head(10)
print(top_countries)

# QUERY 3: Content Added by Year
print("\n3️⃣ CONTENT ADDED BY YEAR")
print("-" * 40)
yearly_content = netflix_df.groupby(['year_added', 'type']).size().reset_index(name='count')
yearly_content = yearly_content.sort_values('year_added', ascending=False)
print(yearly_content.head(20))

# QUERY 4: Top 10 Genres
print("\n4️⃣ TOP 10 GENRES")
print("-" * 40)
genres = netflix_df['listed_in'].str.split(',', expand=True).stack()
genres = genres.str.strip()
top_genres = genres.value_counts().head(10)
print(top_genres)

# QUERY 5: Rating Distribution
print("\n5️⃣ RATING DISTRIBUTION")
print("-" * 40)
rating_dist = netflix_df['rating'].value_counts()
print(rating_dist.head(10))

# QUERY 6: Top 10 Directors
print("\n6️⃣ TOP 10 DIRECTORS")
print("-" * 40)
directors = netflix_df[netflix_df['director'] != 'Unknown']['director']
directors = directors.str.split(',', expand=True).stack()
directors = directors.str.strip()
top_directors = directors.value_counts().head(10)
print(top_directors)

# QUERY 7: Content by Release Year (Decade)
print("\n7️⃣ CONTENT BY DECADE")
print("-" * 40)
netflix_df['decade'] = (netflix_df['release_year'] // 10) * 10
decade_content = netflix_df.groupby(['decade', 'type']).size().reset_index(name='count')
decade_content = decade_content.sort_values('decade', ascending=False)
print(decade_content.head(20))

# QUERY 8: Monthly Addition Pattern
print("\n8️⃣ MONTHLY CONTENT ADDITION PATTERN")
print("-" * 40)
monthly_pattern = netflix_df.groupby('month_name').size().reset_index(name='count')
# Sort by month order
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_pattern['month_name'] = pd.Categorical(monthly_pattern['month_name'],
                                                categories=month_order,
                                                ordered=True)
monthly_pattern = monthly_pattern.sort_values('month_name')
print(monthly_pattern)

# QUERY 9: Average Duration by Type
print("\n9️⃣ AVERAGE DURATION BY TYPE")
print("-" * 40)
avg_duration = netflix_df.groupby('type')['duration_int'].mean().round(2)
print(avg_duration)

# QUERY 10: Recent Additions (Last 2 Years)
print("\n🔟 RECENT ADDITIONS (LAST 2 YEARS)")
print("-" * 40)
recent_date = datetime.now() - pd.DateOffset(years=2)
recent_content = netflix_df[netflix_df['date_added'] >= recent_date]
print(f"Total recent additions: {len(recent_content)}")
print(recent_content[['title', 'type', 'date_added', 'rating']].head(10))

# =============================================
# STEP 6: KEY METRICS (KPIs)
# =============================================

print("\n" + "=" * 60)
print("📊 KEY PERFORMANCE INDICATORS")
print("=" * 60)

total_content = len(netflix_df)
total_movies = len(netflix_df[netflix_df['type'] == 'Movie'])
total_tv_shows = len(netflix_df[netflix_df['type'] == 'TV Show'])
movie_percentage = round((total_movies / total_content) * 100, 2)
tv_show_percentage = round((total_tv_shows / total_content) * 100, 2)
total_countries = netflix_df['country'].nunique()
total_genres = genres.nunique()
avg_movie_duration = netflix_df[netflix_df['type'] == 'Movie']['duration_int'].mean()

print(f"📺 Total Content: {total_content}")
print(f"🎬 Total Movies: {total_movies} ({movie_percentage}%)")
print(f"📺 Total TV Shows: {total_tv_shows} ({tv_show_percentage}%)")
print(f"🌍 Countries Represented: {total_countries}")
print(f"🎭 Total Genres: {total_genres}")
print(f"⏱️  Average Movie Duration: {avg_movie_duration:.0f} minutes")

# =============================================
# STEP 7: DATA VISUALIZATIONS
# =============================================

print("\n" + "=" * 60)
print("📈 CREATING VISUALIZATIONS")
print("=" * 60)

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# VISUALIZATION 1: Content Type Distribution
print("\n📊 Creating Visualization 1: Content Type Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#E50914', '#564d4d']
content_type.plot(kind='pie', autopct='%1.1f%%', colors=colors,
                  startangle=90, ax=ax, textprops={'fontsize': 12})
ax.set_ylabel('')
ax.set_title('Netflix Content Distribution: Movies vs TV Shows', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# VISUALIZATION 2: Top 10 Countries
print("\n📊 Creating Visualization 2: Top 10 Countries")
fig, ax = plt.subplots(figsize=(12, 6))
top_countries.plot(kind='barh', color='#E50914', ax=ax)
ax.set_xlabel('Number of Titles', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
ax.set_title('Top 10 Countries Producing Netflix Content', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# VISUALIZATION 3: Content Added Over Years
print("\n📊 Creating Visualization 3: Content Added Over Years")
yearly_pivot = yearly_content.pivot(index='year_added', columns='type', values='count').fillna(0)
fig, ax = plt.subplots(figsize=(14, 6))
yearly_pivot.plot(kind='line', marker='o', linewidth=2, ax=ax)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.set_title('Netflix Content Addition Trend Over Years', fontsize=16, fontweight='bold')
ax.legend(title='Type', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# VISUALIZATION 4: Top 10 Genres
print("\n📊 Creating Visualization 4: Top 10 Genres")
fig, ax = plt.subplots(figsize=(12, 6))
top_genres.plot(kind='barh', color='#E50914', ax=ax)
ax.set_xlabel('Number of Titles', fontsize=12)
ax.set_ylabel('Genre', fontsize=12)
ax.set_title('Top 10 Most Popular Genres on Netflix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# VISUALIZATION 5: Rating Distribution
print("\n📊 Creating Visualization 5: Rating Distribution")
fig, ax = plt.subplots(figsize=(12, 6))
rating_dist.head(10).plot(kind='bar', color='#E50914', ax=ax)
ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.set_title('Content Distribution by Rating', fontsize=16, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# VISUALIZATION 6: Monthly Addition Pattern
print("\n📊 Creating Visualization 6: Monthly Addition Pattern")
fig, ax = plt.subplots(figsize=(14, 6))
monthly_pattern.plot(x='month_name', y='count', kind='bar', color='#E50914', ax=ax, legend=False)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Titles Added', fontsize=12)
ax.set_title('Monthly Content Addition Pattern', fontsize=16, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# VISUALIZATION 7: Content by Decade
print("\n📊 Creating Visualization 7: Content by Decade")
decade_pivot = decade_content.pivot(index='decade', columns='type', values='count').fillna(0)
fig, ax = plt.subplots(figsize=(14, 6))
decade_pivot.plot(kind='bar', stacked=True, color=['#E50914', '#564d4d'], ax=ax)
ax.set_xlabel('Decade', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.set_title('Netflix Content Distribution by Release Decade', fontsize=16, fontweight='bold')
ax.legend(title='Type', fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# =============================================
# STEP 8: EXPORT RESULTS FOR POWER BI
# =============================================

print("\n" + "=" * 60)
print("💾 EXPORTING DATA FOR POWER BI")
print("=" * 60)

# Export cleaned data
netflix_df.to_csv('netflix_cleaned_data.csv', index=False)
print("✅ Exported: netflix_cleaned_data.csv")

# Export aggregated tables
content_distribution.to_csv('content_type_distribution.csv')
print("✅ Exported: content_type_distribution.csv")

top_countries.to_csv('top_countries.csv')
print("✅ Exported: top_countries.csv")

yearly_content.to_csv('yearly_content.csv', index=False)
print("✅ Exported: yearly_content.csv")

top_genres.to_csv('top_genres.csv')
print("✅ Exported: top_genres.csv")

monthly_pattern.to_csv('monthly_pattern.csv', index=False)
print("✅ Exported: monthly_pattern.csv")

# =============================================
# STEP 9: DOWNLOAD ALL FILES
# =============================================

print("\n" + "=" * 60)
print("📥 DOWNLOAD FILES TO YOUR COMPUTER")
print("=" * 60)

from google.colab import files

# Download all exported files
files.download('netflix_cleaned_data.csv')
files.download('content_type_distribution.csv')
files.download('top_countries.csv')
files.download('yearly_content.csv')
files.download('top_genres.csv')
files.download('monthly_pattern.csv')

print("\n✅ All files downloaded successfully!")

# =============================================
# STEP 10: KEY INSIGHTS SUMMARY
# =============================================

print("\n" + "=" * 60)
print("💡 KEY INSIGHTS & FINDINGS")
print("=" * 60)

print(f"""
1. CONTENT MIX
   • Movies represent {movie_percentage}% of Netflix's catalog
   • TV Shows make up {tv_show_percentage}% of the content

2. GEOGRAPHIC DISTRIBUTION
   • Top content-producing country: {top_countries.index[0]}
   • Total countries represented: {total_countries}

3. GENRE TRENDS
   • Most popular genre: {top_genres.index[0]}
   • Total unique genres: {total_genres}

4. CONTENT DURATION
   • Average movie duration: {avg_movie_duration:.0f} minutes
   • This is optimized for viewer engagement

5. ADDITION PATTERN
   • Peak month for additions: {monthly_pattern.loc[monthly_pattern['count'].idxmax(), 'month_name']}
   • Strategic timing around holidays and viewership trends

6. RATING FOCUS
   • Most common rating: {rating_dist.index[0]}
   • Netflix targets diverse audience demographics

7. TEMPORAL TRENDS
   • Content additions have {('increased' if yearly_content['count'].iloc[0] > yearly_content['count'].iloc[-1] else 'decreased')} over recent years
   • Focus on {('recent releases' if netflix_df['release_year'].mean() > 2010 else 'classic content')}
""")

print("=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("=" * 60)
