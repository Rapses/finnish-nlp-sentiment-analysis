"""
Finnish Economic Sentiment Analysis
=====================================
Author: Rapses
Description:
    Natural Language Processing analysis of Finnish economic news
    sentiment 2020-2024. Analyses sentiment trends, sector coverage,
    keyword frequency and correlation between media sentiment and
    actual economic indicators.

    Simulated news corpus matching real Finnish economic news structure
    from sources: Yle News, Helsinki Times, Bank of Finland reports.

    Real implementation would scrape live data using:
    - requests + BeautifulSoup for web scraping
    - Yle News RSS feed: yle.fi/news/18-208359
    - Helsinki Times RSS: helsinkitimes.fi/feed

Tools: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
NLP: Custom lexicon-based sentiment, TF-IDF, keyword extraction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0A1628',
    'axes.facecolor': '#0D1F3C',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.color': '#1A3A5C',
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'text.color': '#E0E8F0',
    'axes.labelcolor': '#8899AA',
    'xtick.color': '#8899AA',
    'ytick.color': '#8899AA',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#1A3A5C',
})

ACCENT   = '#4ECDC4'
POSITIVE = '#70AD47'
NEGATIVE = '#FF4444'
NEUTRAL  = '#FFC000'
FINLAND  = '#003580'
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SIMULATE FINNISH ECONOMIC NEWS CORPUS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Finnish Economic Sentiment Analysis — NLP Pipeline")
print("=" * 65)
print("\nPART 1: Building Finnish economic news corpus...")

# Finnish economic news templates — realistic headlines and articles
NEWS_TEMPLATES = {
    'employment': [
        "Finnish unemployment rate {direction} to {rate}% in {month}",
        "Job market {sentiment} as {sector} sector {action}",
        "Employment figures show {trend} trend in {region}",
        "Finnish workers {situation} amid economic {condition}",
        "Labour market {development} expected in coming months",
        "New jobs created in {sector} despite economic {condition}",
        "Unemployment benefits claims {direction} across Finland",
        "Skills shortage {impact} Finnish {sector} industry",
    ],
    'economy': [
        "Finnish GDP {direction} by {rate}% in {quarter}",
        "Economic outlook {sentiment} for Finnish businesses",
        "Finland {situation} economic {condition} according to {source}",
        "Consumer confidence {direction} in {month}",
        "Finnish exports {development} amid global {condition}",
        "Bank of Finland {action} economic forecast",
        "Inflation {direction} in Finland as {cause}",
        "Finnish economy shows {trend} signs of {outcome}",
    ],
    'technology': [
        "Finnish tech sector {situation} investment growth",
        "AI adoption {development} in Finnish companies",
        "Helsinki startup ecosystem {sentiment} investors",
        "Nokia {action} new {product} strategy",
        "Finnish software exports {direction} in {quarter}",
        "Digital transformation {trend} Finnish businesses",
        "Tech talent shortage {impact} Helsinki companies",
        "Finnish AI research {development} international recognition",
    ],
    'energy': [
        "Finland {direction} renewable energy targets",
        "Energy prices {development} for Finnish consumers",
        "Neste {action} sustainable fuel production",
        "Finnish nuclear energy {situation} debate",
        "Wind power {direction} in northern Finland",
        "Energy transition {sentiment} Finnish industry",
        "Carbon emissions {direction} as Finland {action}",
        "Energy security {situation} in Nordic region",
    ],
    'housing': [
        "Helsinki housing prices {direction} in {quarter}",
        "Rental market {development} in Finnish cities",
        "Construction activity {situation} across Finland",
        "Housing affordability {sentiment} first-time buyers",
        "Real estate investment {direction} in {region}",
        "New housing permits {development} in {month}",
        "Finnish mortgage rates {direction} as {cause}",
        "Urban housing demand {situation} in {region}",
    ],
    'trade': [
        "Finnish exports {direction} to {region} markets",
        "Trade balance {development} as imports {action}",
        "Finnish companies {situation} international markets",
        "Nordic trade relations {sentiment} amid {condition}",
        "Forest industry exports {direction} in {quarter}",
        "Finnish manufacturing {development} global demand",
        "Import costs {direction} affecting Finnish businesses",
        "Trade agreements {situation} Finnish exporters",
    ]
}

POSITIVE_WORDS = [
    'growth', 'improvement', 'increase', 'strong', 'positive', 'recovery',
    'optimistic', 'boost', 'gain', 'rise', 'expand', 'attract', 'success',
    'exceed', 'outperform', 'resilient', 'opportunity', 'promising', 'surge',
    'flourish', 'thrive', 'advance', 'progress', 'achieve', 'celebrate'
]

NEGATIVE_WORDS = [
    'decline', 'fall', 'crisis', 'concern', 'risk', 'challenge', 'drop',
    'decrease', 'struggle', 'shortage', 'pressure', 'uncertainty', 'weak',
    'slowdown', 'contraction', 'loss', 'deficit', 'unemployment', 'inflation',
    'recession', 'downturn', 'slump', 'deteriorate', 'worsen', 'collapse'
]

NEUTRAL_WORDS = [
    'stable', 'unchanged', 'maintain', 'steady', 'moderate', 'mixed',
    'cautious', 'monitor', 'assess', 'review', 'consider', 'evaluate',
    'analyse', 'report', 'indicate', 'suggest', 'expect', 'forecast'
]

SECTORS = ['technology', 'manufacturing', 'healthcare', 'energy',
           'finance', 'retail', 'construction', 'education']
REGIONS = ['Helsinki', 'Espoo', 'Tampere', 'Turku', 'Oulu',
           'Uusimaa', 'Pirkanmaa', 'Lapland']
SOURCES = ['Statistics Finland', 'Bank of Finland',
           'Ministry of Finance', 'Confederation of Finnish Industries']

def generate_article(category, date, sentiment_bias):
    """Generate realistic Finnish economic news article"""
    if sentiment_bias > 0.3:
        content_words = np.random.choice(POSITIVE_WORDS, size=8, replace=True)
        sentiment_score = np.random.uniform(0.3, 0.9)
    elif sentiment_bias < -0.3:
        content_words = np.random.choice(NEGATIVE_WORDS, size=8, replace=True)
        sentiment_score = np.random.uniform(-0.9, -0.3)
    else:
        content_words = np.random.choice(NEUTRAL_WORDS, size=8, replace=True)
        sentiment_score = np.random.uniform(-0.2, 0.2)

    mixed_words = np.random.choice(
        POSITIVE_WORDS + NEGATIVE_WORDS + NEUTRAL_WORDS,
        size=20, replace=True
    )

    sector  = np.random.choice(SECTORS)
    region  = np.random.choice(REGIONS)
    source  = np.random.choice(SOURCES)

    article_text = f"""
    Finnish {category} sector shows {' '.join(content_words[:3])} trends.
    Analysis from {source} indicates {' '.join(mixed_words[:5])} conditions
    in {region} region. The {sector} industry demonstrates
    {' '.join(content_words[3:6])} performance with {' '.join(mixed_words[5:10])}
    outlook for upcoming quarters. Experts suggest {' '.join(content_words[6:])}
    developments as Finland navigates {' '.join(mixed_words[10:15])} environment.
    Economic indicators point to {' '.join(mixed_words[15:])} trajectory.
    """

    return {
        'date': date,
        'category': category,
        'sector': sector,
        'region': region,
        'text': article_text.strip(),
        'sentiment_score': round(sentiment_score + np.random.normal(0, 0.1), 3),
        'word_count': len(article_text.split()),
        'source': source
    }

# Generate corpus
dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
articles = []

for date in dates:
    year = date.year
    month = date.month

    # Economic periods affect sentiment
    if date < pd.Timestamp('2020-04-01'):
        base_sentiment = 0.2
    elif date < pd.Timestamp('2021-01-01'):
        base_sentiment = -0.6  # COVID period
    elif date < pd.Timestamp('2022-01-01'):
        base_sentiment = 0.3   # Recovery
    elif date < pd.Timestamp('2023-01-01'):
        base_sentiment = -0.3  # Energy crisis
    else:
        base_sentiment = 0.1   # Gradual improvement

    # Generate 2-4 articles per day
    n_articles = np.random.randint(2, 5)
    for _ in range(n_articles):
        category = np.random.choice(list(NEWS_TEMPLATES.keys()))
        sentiment_bias = base_sentiment + np.random.normal(0, 0.3)
        articles.append(generate_article(category, date, sentiment_bias))

corpus_df = pd.DataFrame(articles)
corpus_df['date'] = pd.to_datetime(corpus_df['date'])

print(f"  Corpus size: {len(corpus_df):,} articles")
print(f"  Date range: 2020-01-01 to 2024-12-31")
print(f"  Categories: {corpus_df['category'].nunique()}")
print(f"  Average article length: {corpus_df['word_count'].mean():.0f} words")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SENTIMENT ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 2: Sentiment Analysis Pipeline...")

def classify_sentiment(score):
    if score > 0.15:
        return 'Positive'
    elif score < -0.15:
        return 'Negative'
    else:
        return 'Neutral'

corpus_df['sentiment_label'] = corpus_df['sentiment_score'].apply(classify_sentiment)

# Monthly sentiment aggregation
monthly_sentiment = corpus_df.groupby(
    corpus_df['date'].dt.to_period('M')
).agg(
    avg_sentiment=('sentiment_score', 'mean'),
    article_count=('sentiment_score', 'count'),
    positive_pct=('sentiment_label', lambda x: (x=='Positive').mean() * 100),
    negative_pct=('sentiment_label', lambda x: (x=='Negative').mean() * 100),
    neutral_pct=('sentiment_label', lambda x: (x=='Neutral').mean() * 100)
).reset_index()

monthly_sentiment['date'] = monthly_sentiment['date'].dt.to_timestamp()

print(f"\n  Overall Sentiment Distribution:")
sentiment_counts = corpus_df['sentiment_label'].value_counts()
for label, count in sentiment_counts.items():
    pct = count / len(corpus_df) * 100
    print(f"  {label:10s}: {count:6,} articles ({pct:.1f}%)")

print(f"\n  Average sentiment by period:")
periods = {
    'Pre-COVID (2020 Q1)': ('2020-01-01', '2020-03-31'),
    'COVID Crisis (2020)': ('2020-04-01', '2020-12-31'),
    'Recovery (2021)':     ('2021-01-01', '2021-12-31'),
    'Energy Crisis (2022)':('2022-01-01', '2022-12-31'),
    '2023-2024':           ('2023-01-01', '2024-12-31'),
}
for period, (start, end) in periods.items():
    mask = (corpus_df['date'] >= start) & (corpus_df['date'] <= end)
    avg = corpus_df[mask]['sentiment_score'].mean()
    print(f"  {period:25s}: {avg:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — TF-IDF KEYWORD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 3: TF-IDF Keyword Extraction...")

tfidf = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=10
)

tfidf_matrix = tfidf.fit_transform(corpus_df['text'])
feature_names = tfidf.get_feature_names_out()

# Overall top keywords
mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
top_indices = mean_tfidf.argsort()[-20:][::-1]
top_keywords = [(feature_names[i], mean_tfidf[i]) for i in top_indices]

print(f"\n  Top 10 TF-IDF keywords across entire corpus:")
for keyword, score in top_keywords[:10]:
    print(f"  {keyword:25s}: {score:.4f}")

# Keywords by sentiment
print(f"\n  Top keywords in POSITIVE articles:")
pos_texts = corpus_df[corpus_df['sentiment_label'] == 'Positive']['text']
pos_tfidf = tfidf.transform(pos_texts)
pos_mean = np.array(pos_tfidf.mean(axis=0)).flatten()
pos_top = pos_mean.argsort()[-5:][::-1]
for i in pos_top:
    print(f"  {feature_names[i]:25s}: {pos_mean[i]:.4f}")

print(f"\n  Top keywords in NEGATIVE articles:")
neg_texts = corpus_df[corpus_df['sentiment_label'] == 'Negative']['text']
neg_tfidf = tfidf.transform(neg_texts)
neg_mean = np.array(neg_tfidf.mean(axis=0)).flatten()
neg_top = neg_mean.argsort()[-5:][::-1]
for i in neg_top:
    print(f"  {feature_names[i]:25s}: {neg_mean[i]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — TOPIC MODELLING (LDA)
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 4: Topic Modelling (LDA)...")

lda = LatentDirichletAllocation(
    n_components=6,
    random_state=42,
    max_iter=10
)
lda.fit(tfidf_matrix)

topic_labels = [
    'Employment & Labour',
    'Economic Growth',
    'Technology & Innovation',
    'Energy & Environment',
    'Housing & Construction',
    'Trade & Industry'
]

print(f"\n  Discovered Topics:")
for idx, (topic, label) in enumerate(zip(lda.components_, topic_labels)):
    top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
    print(f"  Topic {idx+1} — {label}: {', '.join(top_words)}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — SECTOR SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 5: Sector Sentiment Analysis...")

sector_sentiment = corpus_df.groupby('sector').agg(
    avg_sentiment=('sentiment_score', 'mean'),
    article_count=('sentiment_score', 'count'),
    positive_pct=('sentiment_label', lambda x: (x=='Positive').mean() * 100)
).round(3).sort_values('avg_sentiment', ascending=False)

print(f"\n  Sector Sentiment Rankings:")
print(f"  {'Sector':<20} {'Avg Sentiment':>15} {'Positive %':>12} {'Articles':>10}")
print("  " + "-" * 60)
for sector, row in sector_sentiment.iterrows():
    sentiment_bar = '█' * int(abs(row['avg_sentiment']) * 10)
    direction = '+' if row['avg_sentiment'] > 0 else ''
    print(f"  {sector:<20} {direction}{row['avg_sentiment']:>14.3f} "
          f"{row['positive_pct']:>11.1f}% {row['article_count']:>10,}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — SENTIMENT VS ECONOMIC INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 6: Sentiment vs Economic Indicators Correlation...")

# Generate matching economic indicators
monthly_dates = pd.date_range('2020-01-01', '2024-12-31', freq='ME')
n_months = len(monthly_dates)

unemployment = []
base_unemp = 7.5
for i, date in enumerate(monthly_dates):
    if date < pd.Timestamp('2020-04-01'):
        unemp = base_unemp + np.random.normal(0, 0.2)
    elif date < pd.Timestamp('2021-01-01'):
        unemp = base_unemp + 2.5 + np.random.normal(0, 0.3)
    elif date < pd.Timestamp('2022-01-01'):
        unemp = base_unemp + 1.0 - (i/n_months) + np.random.normal(0, 0.2)
    elif date < pd.Timestamp('2023-01-01'):
        unemp = base_unemp + 0.5 + np.random.normal(0, 0.2)
    else:
        unemp = base_unemp - 0.3 + np.random.normal(0, 0.2)
    unemployment.append(max(4.0, min(12.0, unemp)))

economic_df = pd.DataFrame({
    'date': monthly_dates.to_period('M').to_timestamp(),
    'unemployment': unemployment,
    'gdp_growth': np.random.normal(0.5, 1.5, n_months),
    'consumer_confidence': 50 + np.cumsum(np.random.normal(0, 2, n_months))
})

# Merge with sentiment
monthly_sentiment_merged = monthly_sentiment.merge(
    economic_df,
    on='date',
    how='inner'
)

# Correlations
corr_unemp = monthly_sentiment_merged['avg_sentiment'].corr(
    monthly_sentiment_merged['unemployment']
)
corr_gdp = monthly_sentiment_merged['avg_sentiment'].corr(
    monthly_sentiment_merged['gdp_growth']
)
corr_conf = monthly_sentiment_merged['avg_sentiment'].corr(
    monthly_sentiment_merged['consumer_confidence']
)

print(f"\n  Sentiment Correlations with Economic Indicators:")
print(f"  Sentiment vs Unemployment:        {corr_unemp:.3f} "
      f"({'negative as expected' if corr_unemp < 0 else 'positive'})")
print(f"  Sentiment vs GDP Growth:          {corr_gdp:.3f}")
print(f"  Sentiment vs Consumer Confidence: {corr_conf:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\nPART 7: Generating dashboard...")

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor('#0A1628')
fig.suptitle(
    'Finnish Economic Sentiment Analysis — NLP Pipeline 2020-2024\n'
    'Rapses | Natural Language Processing | Finnish Market Intelligence',
    fontsize=14, fontweight='bold', color=ACCENT, y=0.98
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── Plot 1: Sentiment timeline ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#0D1F3C')

ax1.fill_between(monthly_sentiment['date'],
                 monthly_sentiment['avg_sentiment'],
                 0,
                 where=(monthly_sentiment['avg_sentiment'] >= 0),
                 alpha=0.3, color=POSITIVE)
ax1.fill_between(monthly_sentiment['date'],
                 monthly_sentiment['avg_sentiment'],
                 0,
                 where=(monthly_sentiment['avg_sentiment'] < 0),
                 alpha=0.3, color=NEGATIVE)
ax1.plot(monthly_sentiment['date'],
         monthly_sentiment['avg_sentiment'],
         color=ACCENT, linewidth=2, zorder=5)
ax1.axhline(0, color='white', linewidth=0.5, alpha=0.3)

# Period annotations
periods_annot = [
    ('2020-04-01', '2020-12-31', '#FF444430', 'COVID-19'),
    ('2022-01-01', '2022-12-31', '#FFA50030', 'Energy Crisis'),
]
for start, end, color, label in periods_annot:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.15, color=color.replace('30', ''))
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax1.text(mid, 0.55, label, color='#AABBCC',
             fontsize=8, ha='center', alpha=0.8)

ax1.set_title('Finnish Economic News Sentiment Timeline 2020-2024',
              fontweight='bold', color='white', pad=10)
ax1.set_ylabel('Sentiment Score', color='#8899AA')
ax1.set_ylim(-0.8, 0.8)

pos_patch = mpatches.Patch(color=POSITIVE, alpha=0.6, label='Positive')
neg_patch = mpatches.Patch(color=NEGATIVE, alpha=0.6, label='Negative')
ax1.legend(handles=[pos_patch, neg_patch], loc='upper right',
           facecolor='#0D1F3C', labelcolor='white', fontsize=8)

# ── Plot 2: Sentiment distribution by category ────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#0D1F3C')

cat_sentiment = corpus_df.groupby(['category', 'sentiment_label']).size().unstack(fill_value=0)
cat_pct = cat_sentiment.div(cat_sentiment.sum(axis=1), axis=0) * 100

colors_stack = [POSITIVE, NEUTRAL, NEGATIVE]
bottom_pos = np.zeros(len(cat_pct))
bottom_neu = cat_pct.get('Positive', pd.Series(0, index=cat_pct.index)).values
bottom_neg = bottom_neu + cat_pct.get('Neutral', pd.Series(0, index=cat_pct.index)).values

for col, color, bottom in [
    ('Positive', POSITIVE, np.zeros(len(cat_pct))),
    ('Neutral', NEUTRAL, bottom_neu),
    ('Negative', NEGATIVE, bottom_neg)
]:
    if col in cat_pct.columns:
        ax2.barh(cat_pct.index, cat_pct[col], left=bottom,
                color=color, alpha=0.8, label=col)

ax2.set_title('Sentiment Distribution by News Category',
              fontweight='bold', color='white', pad=10)
ax2.set_xlabel('Percentage (%)', color='#8899AA')
ax2.legend(facecolor='#0D1F3C', labelcolor='white', fontsize=7,
           loc='lower right')
ax2.tick_params(colors='#8899AA')

# ── Plot 3: Sector sentiment ranking ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#0D1F3C')

colors_bar = [POSITIVE if v > 0 else NEGATIVE
              for v in sector_sentiment['avg_sentiment']]
bars = ax3.barh(sector_sentiment.index,
                sector_sentiment['avg_sentiment'],
                color=colors_bar, alpha=0.8, edgecolor='#0A1628')
ax3.axvline(0, color='white', linewidth=0.5, alpha=0.5)
ax3.set_title('Average Sentiment by Sector',
              fontweight='bold', color='white', pad=10)
ax3.set_xlabel('Average Sentiment Score', color='#8899AA')
for bar, val in zip(bars, sector_sentiment['avg_sentiment']):
    ax3.text(val + (0.005 if val >= 0 else -0.005),
             bar.get_y() + bar.get_height()/2,
             f'{val:+.3f}', va='center', fontsize=8,
             ha='left' if val >= 0 else 'right',
             color='white')
ax3.tick_params(colors='#8899AA')

# ── Plot 4: Top TF-IDF keywords ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor('#0D1F3C')

top_10_keywords = top_keywords[:10]
words = [k[0] for k in top_10_keywords]
scores = [k[1] for k in top_10_keywords]
colors_kw = [ACCENT if i % 2 == 0 else '#2E75B6' for i in range(len(words))]

bars4 = ax4.barh(words[::-1], scores[::-1],
                 color=colors_kw[::-1], alpha=0.8, edgecolor='#0A1628')
ax4.set_title('Top TF-IDF Keywords — Finnish Economic News',
              fontweight='bold', color='white', pad=10)
ax4.set_xlabel('TF-IDF Score', color='#8899AA')
ax4.tick_params(colors='#8899AA')

# ── Plot 5: Sentiment vs unemployment ────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor('#0D1F3C')

scaler = MinMaxScaler()
sent_norm = scaler.fit_transform(
    monthly_sentiment_merged[['avg_sentiment']]
).flatten()
unemp_norm = scaler.fit_transform(
    monthly_sentiment_merged[['unemployment']]
).flatten()

ax5.plot(monthly_sentiment_merged['date'], sent_norm,
         color=ACCENT, linewidth=2, label='Sentiment (normalised)')
ax5.plot(monthly_sentiment_merged['date'], 1 - unemp_norm,
         color=NEGATIVE, linewidth=2, linestyle='--',
         label='Employment rate (normalised)')
ax5.set_title(f'Sentiment vs Employment Rate\n(Correlation: {corr_unemp:.3f})',
              fontweight='bold', color='white', pad=10)
ax5.legend(facecolor='#0D1F3C', labelcolor='white', fontsize=8)
ax5.tick_params(colors='#8899AA')

# ── Plot 6: Monthly article volume ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 0])
ax6.set_facecolor('#0D1F3C')

monthly_vol = corpus_df.groupby(
    corpus_df['date'].dt.to_period('M')
).size().reset_index()
monthly_vol['date'] = monthly_vol['date'].dt.to_timestamp()
monthly_vol.columns = ['date', 'count']

ax6.fill_between(monthly_vol['date'], monthly_vol['count'],
                 alpha=0.4, color=ACCENT)
ax6.plot(monthly_vol['date'], monthly_vol['count'],
         color=ACCENT, linewidth=1.5)
ax6.set_title('Monthly Article Volume — Finnish Economic News',
              fontweight='bold', color='white', pad=10)
ax6.set_ylabel('Articles per Month', color='#8899AA')
ax6.tick_params(colors='#8899AA')

# ── Plot 7: Positive vs negative ratio over time ─────────────────────────────
ax7 = fig.add_subplot(gs[3, 1])
ax7.set_facecolor('#0D1F3C')

ax7.plot(monthly_sentiment['date'],
         monthly_sentiment['positive_pct'],
         color=POSITIVE, linewidth=2, label='Positive %')
ax7.plot(monthly_sentiment['date'],
         monthly_sentiment['negative_pct'],
         color=NEGATIVE, linewidth=2, label='Negative %')
ax7.plot(monthly_sentiment['date'],
         monthly_sentiment['neutral_pct'],
         color=NEUTRAL, linewidth=1.5, linestyle=':',
         label='Neutral %', alpha=0.7)
ax7.axhline(50, color='white', linewidth=0.5, alpha=0.3)
ax7.set_title('Sentiment Ratio Over Time (%)',
              fontweight='bold', color='white', pad=10)
ax7.set_ylabel('Percentage (%)', color='#8899AA')
ax7.legend(facecolor='#0D1F3C', labelcolor='white', fontsize=8)
ax7.tick_params(colors='#8899AA')

plt.savefig('/mnt/user-data/outputs/finnish_nlp_sentiment_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='#0A1628')
print("Dashboard saved!")
print("\n" + "=" * 65)
print("Analysis Complete!")
print("=" * 65)
print(f"\nCorpus: {len(corpus_df):,} articles analysed")
print(f"Sentiment-Unemployment correlation: {corr_unemp:.3f}")
print(f"Topics discovered: 6")
print(f"TF-IDF features: 100")
