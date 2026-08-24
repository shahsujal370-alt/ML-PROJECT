**-----ML-Based Loan Pricing Fairness & Anomaly Detection-----**

 1. Project objective

To build an end-to-end machine-learning monitoring system that estimates an expected loan interest rate from borrower and loan characteristics, then compares the observed rate with the model benchmark to identify potentially unusual pricing. The project combines supervised regression, residual analysis, peer-group benchmarking, unsupervised anomaly detection, statistical fairness monitoring, and explainability.

 2. Dataset

The project uses the supplied loan CSV. It contains 32,586 rows and 13 columns. The observed pricing target is loan_int_rate.

Important fields:
-  customer_income
-  loan_amnt
-  customer_age
-  loan_grade
-  term_years
-  historical_default
-  cred_hist_length
-  employment_duration
-  home_ownership
-  loan_intent
-  loan_int_rate
-  Current_loan_status

customer_id is an identifier and is removed. Current_loan_status is excluded from the primary pricing model to avoid outcome leakage.

 3. Business question

> Given observable borrower and loan characteristics, what interest rate would the model expect for a loan, and which observed rates are unusually far from that benchmark or from comparable peer loans?

 4. Methodology

A. Data preparation
1. Parse currency-formatted income and loan amount.
2. Remove records with missing target  loan_int_rate from the regression sample.
3. Impute missing numerical values with the median.
4. Treat missing historical-default values as Unknown.
5. One-hot encode categorical variables.
6. Exclude customer ID and current loan outcome from the pricing model.

 B. Model development
Models compared:
- Linear Regression baseline
- Random Forest Regression
- XGBoost 

The best model is selected using test-set RMSE in the supplied reproducible experiment.

 C. Pricing residual

 pricing_residual = actual_interest_rate - expected_interest_rate

Positive residual: observed pricing is higher than the model benchmark.
Negative residual: observed pricing is lower than the benchmark.

 D. Pricing anomaly detection

The project flags the top 1% of absolute pricing residuals and also reports observations beyond ±3 residual standard deviations.

 E. Profile anomaly detection

Isolation Forest is applied to borrower/loan numeric characteristics to identify unusual profiles independent of the pricing residual.

 F. Peer-group analysis

Loans are grouped using loan grade, term, income band and loan-amount band. Each loan is compared with its peer-group median rate. Peer groups with fewer than 10 observations are not used for peer anomaly flags.

 G. Fairness monitoring

Pricing residuals and anomaly rates are summarized across age groups, home ownership, loan grade and loan intent. Welch's t-tests are included for selected high-volume group comparisons. These tests are monitoring diagnostics, not legal conclusions.

 H. Explainability

For the portfolio version, SHAP can be added to explain the fitted tree model. The model's explanation describes why the ML model expects a particular rate; it does not prove why a lender actually charged that rate.

5. Results from the supplied dataset:

After cleaning, 29,470 loans had a non-missing interest-rate target. The experiment used 23,576 training observations and 5,894 test observations.

| Model | MAE | RMSE | R² |

| Linear Regression | 1.356 | 1.698 | 0.729 |
| Random Forest | 1.163 | 1.521 | 0.783 |
| XGBoost benchmark | 1.186 | 1.531 | 0.780 |

The Random Forest was selected because it had the lowest test RMSE in this experiment.

 ---Anomaly results:---
- Absolute residual 99th-percentile threshold: 4.11 percentage points
- Pricing residual anomalies: 59 / 5,894 (1.00%)
- ±3 residual-SD observations: 23
- Isolation Forest profile anomalies: 59 / 5,894 (1.00%)
- Peer-group anomalies: 34
- Overlap between pricing and peer anomalies: 11

---Fairness monitoring result:---
The mean residual for age 18–25 was approximately -0.046 percentage points versus approximately 0.000 percentage points for age 26–35; Welch's t-test p-value was about 0.269. For MORTGAGE versus RENT, the mean residuals were approximately -0.039 and -0.009 percentage points respectively; p-value was about 0.470.

These results do not provide evidence of a material residual difference in those two selected comparisons in this test sample. They also do not prove that the overall pricing process is fair.

