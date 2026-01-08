This comprehensive report examines a sophisticated deep learning framework developed for forecasting stock price dynamics and sectoral trends in the Indian equity market. The research employs Long Short-Term Memory (LSTM) neural networks to analyze daily adjusted closing prices from 2015-2021 for the BSE Sensex and representative sectoral equities across five key sectors: Energy, IT Services, Banking, Pharmaceuticals, and FMCG.
The study distinguishes itself by prioritizing interpretability, robustness across market regimes, and probabilistic assessment of forecast reliability over mere point prediction accuracy. Key findings reveal that LSTM architectures effectively capture medium-term trend persistence but exhibit performance degradation during abrupt regime shifts, particularly during the COVID-19 market shock. The integration of ensemble learning and Bayesian uncertainty quantification through Monte Carlo Dropout significantly enhances the practical utility of these models for financial decision-making.
Key Outcomes:
•	Directional accuracy remains modest but statistically stable across all sectors
•	Ensemble averaging reduces prediction volatility by incorporating multiple model perspectives
•	Strong trend-tracking capabilities during stable market periods with systematic lag during volatile periods
•	Monte Carlo uncertainty bands provide valuable reliability signals, expanding significantly during crisis periods
•	Sectoral heterogeneity reveals defensive sectors exhibit lower volatility while cyclical sectors display greater forecast dispersion
[yahoo_finance_data_sources_2015_2021.csv](https://github.com/user-attachments/files/24494928/yahoo_finance_data_sources_2015_2021.csv)Name,Ticker,URL
BSE Sensex,^BSESN,https://finance.yahoo.com/quote/%5EBSESN/history?period1=1420070400&period2=1640908800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
Reliance Industries,RELIANCE.NS,https://finance.yahoo.com/quote/RELIANCE.NS/history?period1=1420070400&period2=1640908800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
Tata Consultancy Services,TCS.NS,https://finance.yahoo.com/quote/TCS.NS/history?period1=1420070400&period2=1640908800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
HDFC Bank,HDFCBANK.NS,https://finance.yahoo.com/quote/HDFCBANK.NS/history?period1=1420070400&period2=1640908800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
Sun Pharmaceutical Industries,SUNPHARMA.NS,https://finance.yahoo.com/quote/SUNPHARMA.NS/history?period1=1420070400&period2=1640908800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
ITC Limited,ITC.NS,https://finance.yahoo.com/quote/ITC.NS/history?period1=1420070400&period2=1640908800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true

