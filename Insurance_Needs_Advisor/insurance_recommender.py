import pandas as pd

class InsuranceRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.clean_data()

    def clean_data(self):
        # Normalize ages
        self.df["Min Age"] = self.df["Min Age"].apply(self._parse_age)
        self.df["Max Age"] = self.df["Max Age"].apply(self._parse_age)
        # Clean premium strings
        self.df["Premium Num"] = self.df["Annual Premium"].apply(self._parse_premium)

    def _parse_age(self, age):
        if pd.isna(age):
            return None
        age = str(age).strip().lower()
        if age in ["all ages", "unknown"]:
            return None
        try:
            return int(age)
        except ValueError:
            return None

    def _parse_premium(self, premium):
        if pd.isna(premium):
            return None
        p = str(premium).lower()
        if any(x in p for x in ["free", "0", "~0", "0 (govt funded)", "0 (bundled)"]):
            return 0
        p = p.replace("~", "").replace("?", "").replace(",", "")
        try:
            return float(p)
        except:
            return None

    def _income_limit_from_notes(self, notes):
        """
        Very naive parser: if notes say 'income <= 100000', extract 100000
        """
        if not notes or pd.isna(notes):
            return None
        text = str(notes).lower()
        if "<=" in text:
            try:
                num = ''.join(filter(str.isdigit, text))
                return int(num)
            except:
                return None
        return None

    def filter_policies(self, user_profile):
        age = user_profile["age"]
        income = user_profile["income"]
        needs = user_profile["needs"]
        max_premium = user_profile.get("max_premium", None)

        filtered = []
        for _, row in self.df.iterrows():
            # Age filter
            if row["Min Age"] and age < row["Min Age"]:
                continue
            if row["Max Age"] and age > row["Max Age"]:
                continue

            # Need / Type match
            if not any(need.lower() in str(row["Type"]).lower() for need in needs):
                continue

            # Premium filter
            premium = row["Premium Num"]
            if premium is not None and max_premium is not None and premium > max_premium:
                continue

            # Income filter from eligibility notes
            income_limit = self._income_limit_from_notes(row.get("Eligibility Notes", ""))
            if income_limit and income > income_limit:
                continue

            filtered.append(row)

        return pd.DataFrame(filtered)

    def score_policy(self, user_profile, row):
        score = 0

        # Prefer very low or free premium
        premium = row["Premium Num"]
        if premium == 0:
            score += 20
        elif premium and premium <= user_profile["income"] * 0.01:
            score += 10
        elif premium and premium <= user_profile["income"] * 0.05:
            score += 5

        # Higher sum assured better
        sum_assured = str(row["Sum Assured"])
        try:
            sum_val = int(''.join(filter(str.isdigit, sum_assured)))
            if sum_val >= 500000:
                score += 15
            elif sum_val >= 200000:
                score += 10
            elif sum_val >= 100000:
                score += 5
        except:
            pass

        return score

    def recommend(self, user_profile, top_n=5):
        filtered_df = self.filter_policies(user_profile)
        if filtered_df.empty:
            return pd.DataFrame()

        # Add scores
        filtered_df = filtered_df.copy()
        filtered_df["Score"] = filtered_df.apply(
            lambda row: self.score_policy(user_profile, row), axis=1
        )
        recommendations = filtered_df.sort_values(
            by="Score", ascending=False
        ).head(top_n)
        return recommendations
