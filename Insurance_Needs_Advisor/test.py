
from insurance_recommender import InsuranceRecommender
recommender = InsuranceRecommender("Insurance_Needs_Advisor/policies.csv")

user_profile = {
    "age": 35,
    "income": 90000,
    "needs": ["Health", "Term Life"],
    "max_premium": 2000
}

top_policies = recommender.recommend(user_profile)
print(top_policies[["Name", "Type", "Annual Premium", "Sum Assured", "Eligibility Notes", "Score"]])

