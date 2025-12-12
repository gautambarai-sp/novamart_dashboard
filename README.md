# 🛒 NovaMart Analytics Dashboard

A comprehensive, interactive analytics dashboard for NovaMart - a rapidly growing omnichannel retail company operating across India. This dashboard provides real-time insights into marketing performance, customer behavior, product sales, and AI-powered lead scoring effectiveness.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Dashboard Overview

NovaMart sells **Electronics, Fashion, and Home & Living** products through both online and offline channels. This dashboard serves the leadership team to explore marketing performance, customer behavior, product sales, and the effectiveness of their lead scoring AI model.

### 🎯 Key Features

| Page | Description |
|------|-------------|
| **Executive Summary** | High-level KPIs, revenue trends, conversion funnel, channel attribution, and customer journey visualization |
| **Campaign Performance** | Deep dive into marketing campaigns across channels, regions, and campaign types with correlation analysis |
| **Customer Analytics** | Customer segmentation, lifetime value analysis, churn prediction, and acquisition channel performance |
| **Product Sales** | Category performance, product rankings, profit margins, and regional sales distribution |
| **Lead Scoring AI** | ML model performance metrics, feature importance, learning curves, and conversion analysis |
| **Geographic Insights** | State-wise market analysis, growth patterns, and store performance across India |

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/novamart-dashboard.git
   cd novamart-dashboard
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
novamart-dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore file
└── data/                       # Data directory
    ├── campaign_performance.csv    # Marketing campaign data (5,858 rows)
    ├── customer_data.csv           # Customer information (5,000 customers)
    ├── product_sales.csv           # Product sales data (1,440 rows)
    ├── lead_scoring_results.csv    # Lead scoring AI results (2,000 leads)
    ├── geographic_data.csv         # State-wise metrics (15 states)
    ├── funnel_data.csv             # Conversion funnel stages
    ├── channel_attribution.csv     # Multi-touch attribution data
    ├── correlation_matrix.csv      # Marketing metrics correlation
    ├── customer_journey.csv        # Customer journey paths
    ├── feature_importance.csv      # ML feature importance
    └── learning_curve.csv          # ML model learning curve
```

## 📈 Data Dictionary

### Campaign Performance
| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Campaign date |
| campaign_id | string | Unique campaign identifier |
| campaign_type | string | Brand Awareness, Retargeting, etc. |
| channel | string | Facebook, Google Ads, Instagram, etc. |
| region | string | North, South, East, West, Central |
| impressions | int | Number of ad impressions |
| clicks | int | Number of ad clicks |
| conversions | int | Number of conversions |
| spend | float | Ad spend in INR |
| revenue | float | Revenue generated in INR |
| ctr | float | Click-through rate |
| roas | float | Return on ad spend |

### Customer Data
| Column | Type | Description |
|--------|------|-------------|
| customer_id | string | Unique customer identifier |
| age_group | string | Customer age bracket |
| income_bracket | string | Low, Medium, High, Premium |
| customer_segment | string | Premium, Regular, Budget, New, Churned |
| acquisition_channel | string | How customer was acquired |
| lifetime_value | float | Customer lifetime value in INR |
| churn_probability | float | Predicted churn probability (0-1) |
| satisfaction_score | float | Customer satisfaction (1-5) |

### Lead Scoring Results
| Column | Type | Description |
|--------|------|-------------|
| lead_id | string | Unique lead identifier |
| company_size | string | Small, Medium, Large, Enterprise |
| industry | string | Healthcare, Finance, Technology, etc. |
| actual_converted | int | Whether lead converted (0/1) |
| predicted_probability | float | Model's conversion probability |
| predicted_class | int | Model's prediction (0/1) |

## 🎨 Visualization Types Used

- **Line Charts**: Time-series trends for revenue, spend, and conversions
- **Bar Charts**: Comparative analysis across categories, channels, and regions
- **Pie/Donut Charts**: Distribution and composition analysis
- **Scatter Plots**: Correlation and clustering visualizations
- **Heatmaps**: Correlation matrices and performance matrices
- **Funnel Charts**: Conversion funnel visualization
- **Sankey Diagrams**: Customer journey flow visualization
- **Geographic Maps**: India state-wise performance visualization
- **Confusion Matrix**: ML model performance evaluation

## 🔧 Configuration

### Date Range Filter
The sidebar includes a date range selector that filters campaign data across all relevant pages.

### Page-Specific Filters
- **Campaign Performance**: Channel, Region, Campaign Type
- **Product Sales**: Category, Region, Year
- **Customer Analytics**: Segment-based views

## 💡 Key Business Insights

### Marketing
- Google Ads excels at first-touch attribution (awareness)
- Email dominates last-touch attribution (conversion)
- Retargeting campaigns show highest ROAS

### Customer
- Premium customers have 10x higher LTV than Budget segment
- 25% of active customers have >25% churn probability
- Referral channel produces highest quality customers

### Products
- Electronics leads revenue, Home & Living leads profit margins
- Products rated >4.2 have 35% higher sales
- West region shows strongest growth

### AI Model
- Lead scoring model achieves ~78% accuracy
- Top predictors: Webinar attendance, Form submissions
- Enterprise leads convert at highest rate

## 🚀 Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Click Deploy!

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Built with ❤️ using Streamlit and Plotly**
