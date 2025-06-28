from needs_advisor import get_insurance_recommendation

response = get_insurance_recommendation(
    age="35",
    dependents="2 children, mom 65 years old dad 70 years old",
    income="12 lakh INR annual",
    assets="1 bhk house only",
    health_conditions="dibetes type 1",
    location="Guwahati"
)

print(response)
