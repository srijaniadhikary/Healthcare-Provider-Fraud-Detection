Healthcare Provider Fraud is one of the biggest problems in today’s world. According to the government, the total Medicare spending increased exponentially due to frauds in Medicare claims. Healthcare fraud is an organized crime which involves peers of providers, physicians, beneficiaries acting together to make fraud claims.
Healthcare fraud and abuse take many forms. Some of the most common types of frauds by providers are:
1. Billing for services that were not provided.
2. Duplicate submission of a claim for the same service.
3. Misrepresenting the service provided.
4. Charging for a more complex or expensive service than was actually provided.
5. Billing for a covered service when the service actually provided was not covered.

The goal of this project is to " predict the potentially fraudulent providers " based on the claims filed by them.

Prepared data from four unorganized datasets and performed EDA, data preprocessing including missing value imputation.
* Created features and performed feature selection using different methods including Weight of Evidence and RFE.
* Trained Logistic Regression, Decision Tree, Random Forest and XGBoost models.
* Performed Hyperparameter Tuning using Bias Variance Trade off.
* Based on Precision selected XGBoost Classifier as the best performing model and obtained Precision∼72% in out of time sample.
* Deployed web application on Heroku platform to predict whether a provider is fraudulent or not.

     *App link:* https://healthcare-provider-fraud.herokuapp.com/
