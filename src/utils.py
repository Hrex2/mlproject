def fill_defaults(df):

    defaults = {
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "DistanceFromHome": 10,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "JobSatisfaction": 3,
        "MonthlyRate": 20000,
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df