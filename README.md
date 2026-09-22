# Task 3: Clustering (Unsupervised Learning)

## Description
Segment customers into behavioral groups using K-Means clustering, without
any labeled data — based on their purchase history.

## Dataset
- **File:** `Online Retail.xlsx`
- **Source:** [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Raw shape:** 541,909 transaction rows (one row per item per invoice), 8 columns
- **Time span:** Dec 1, 2010 - Dec 9, 2011, UK-based online retailer (91% of
  transactions from the UK)

Unlike Tasks 1 and 2, this dataset is **transactional**, not one-row-per-entity.
The core challenge of this task was transforming raw transactions into
one row per customer before clustering was even possible.

## Steps Performed

### 1. Exploration
Checked shape, dtypes, missing values, duplicates, and value ranges.
Found: 135,080 rows (25%) missing `CustomerID`, 1,454 missing `Description`,
5,268 duplicate rows, and negative values in both `Quantity` (returns) and
`UnitPrice` (bookkeeping adjustments).

### 2. Cleaning
- Dropped rows with no `CustomerID` (can't segment an unidentified customer)
- Removed exact duplicate rows
- Removed returns/cancellations (`Quantity <= 0`) and invalid prices
  (`UnitPrice <= 0`)
- Result: 541,909 -> 392,692 rows; 4,372 -> 4,338 unique customers retained

### 3. RFM Feature Engineering
Aggregated all transactions into one row per customer using the classic
**RFM** framework:
- **Recency** - days since the customer's last purchase (relative to one day
  after the dataset's last recorded transaction)
- **Frequency** - number of distinct invoices (orders) placed
- **Monetary** - total amount spent (`Quantity x UnitPrice`, summed)

### 4. Outlier Investigation & Log Transformation
RFM values were heavily right-skewed - e.g. Monetary ranged from $3.75 to
$280,206.02 (mean $2,049, median only $669). A handful of wholesale/bulk
buyers (200+ orders, $100K+ spend) would have dominated distance-based
clustering if left untreated.
Applied `log1p()` to Recency, Frequency, and Monetary to compress the scale
and reduce skew before clustering. Monetary became close to normally
distributed; Frequency remained somewhat skewed even after transformation,
reflecting a genuine pattern - most customers are one-time or occasional
buyers.

### 5. Scaling
Applied `StandardScaler` to the three log-transformed features so no single
feature dominates distance calculations due to differing numeric ranges.

### 6. Choosing k
Tested k = 2 to 10 using both the elbow method (inertia) and silhouette
score. Silhouette score was technically highest at k=2 (0.4328), but this
would only split customers into two broad groups with little business
value. The elbow visibly bends at **k=4**, where silhouette score (0.3375)
is essentially tied with k=3 - chosen as the best balance between
statistical cluster quality and actionable business granularity.

### 7. Final Model & Visualization
Fit `KMeans(n_clusters=4)` on the scaled RFM features, then visualized with
PCA (2 components captured **93.87%** of total variance - a highly faithful
2D representation of the actual clustering).

### 8. Interpretation
Mapped each cluster to a human-readable segment label based on its RFM
profile, added as a `Segment` column, and saved the full result to
`rfm_clustered_customers.csv`.

## Results

| Cluster | Segment | Customers | Avg. Recency | Avg. Frequency | Avg. Monetary |
|---|---|---|---|---|---|
| 0 | Champions | 713 (16%) | 12.2 days | 13.8 orders | $8,088 |
| 1 | Lost/Churned | 1,622 (37%) | 181.5 days | 1.3 orders | $341 |
| 2 | New/Recent | 837 (19%) | 17.7 days | 2.2 orders | $557 |
| 3 | Regulars | 1,166 (27%) | 71.6 days | 4.1 orders | $1,802 |

**Final Silhouette Score (k=4): 0.3375**

### Key findings
- **Champions (16%)** are the most valuable segment - recent, frequent
  buyers who spend ~24x more on average than the Lost/Churned group.
  Priority for retention efforts (loyalty rewards, early access).
- **Lost/Churned is the single largest segment (37%)** - customers who
  haven't purchased in ~6 months and rarely ordered even when active. This
  is a significant finding: more customers have disengaged than remain in
  any other single segment, representing either a major win-back
  opportunity or an acceptable/expected churn rate to benchmark against.
- **New/Recent (19%)** customers purchased recently but haven't built up
  frequency or spend yet - a nurture/onboarding opportunity to convert them
  toward Regulars or Champions.
- **Regulars (27%)** sit in the middle on all three RFM dimensions - the
  largest pool of customers with realistic upsell potential into Champions.
- Boxplots of RFM by cluster show that even within well-separated clusters,
  the original (non-log) values remain right-skewed - expected, since
  clustering was performed in log-space while these plots show raw dollar/
  day values.

## Output File
`rfm_clustered_customers.csv` - one row per customer with Recency,
Frequency, Monetary (raw and log-transformed), assigned Cluster number,
PCA coordinates, and a human-readable Segment label.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place `Online Retail.xlsx` in the same folder as `main.py`
3. Run: `python main.py`
4. Output is saved as `rfm_clustered_customers.csv` in the same folder

## Tools
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl (for
reading .xlsx)
