"""Test questions for the data agent, with expected answers computed by pandas.

Each question has:
  id, dataset, question, kind (what it tests)
  numbers  function(df) -> list of numbers that must all appear in the answer
  text     words that must all appear in the answer (case-insensitive)
  chart    True if at least one chart must be made
  shape    function(df) -> (rows, cols) the dataset must have afterwards (cleaning questions)
  tool     a tool that must be used (e.g. explain_change)
  rules    saved user rules given to the agent with the question
  tol      relative tolerance for numbers (default 1%)
"""

DATASETS = {
    "cancer": "cancer-risk-factors.csv",
    "netflix": "netflix_titles.csv",
    "uber": "ncr_ride_bookings.csv",
}


def _march(u):
    return u[u["Date"].str.startswith("2024-03")]


def _mean_by(df, group, col):
    return df.groupby(group)[col].mean()


QUESTIONS = [
    # ---------------------------------------------------------------- cancer
    dict(id="c1", dataset="cancer", kind="lookup",
         question="How many patients are in the dataset?",
         numbers=lambda c: [len(c)]),
    dict(id="c2", dataset="cancer", kind="aggregate",
         question="What is the average age of the patients?",
         numbers=lambda c: [c["Age"].mean()]),
    dict(id="c3", dataset="cancer", kind="group-by",
         question="Which cancer type has the highest average Overall_Risk_Score, and what is that average?",
         text=["Lung"], numbers=lambda c: [_mean_by(c, "Cancer_Type", "Overall_Risk_Score").max()]),
    dict(id="c4", dataset="cancer", kind="filter",
         question="How many patients have a High risk level?",
         numbers=lambda c: [(c["Risk_Level"] == "High").sum()]),
    dict(id="c5", dataset="cancer", kind="filter",
         question="How many patients have a Smoking value above 7?",
         numbers=lambda c: [(c["Smoking"] > 7).sum()]),
    dict(id="c6", dataset="cancer", kind="filter+aggregate",
         question="What is the average BMI of patients with Lung cancer?",
         numbers=lambda c: [c.loc[c["Cancer_Type"] == "Lung", "BMI"].mean()]),
    dict(id="c7", dataset="cancer", kind="chart",
         question="Show a bar chart of the number of patients per cancer type.",
         chart=True),
    dict(id="c8", dataset="cancer", kind="root-cause",
         question="Why is the average Overall_Risk_Score higher for Lung cancer than for Prostate cancer? "
                  "Which factors drive the difference?",
         tool="explain_change",
         numbers=lambda c: [_mean_by(c, "Cancer_Type", "Overall_Risk_Score")["Prostate"],
                            _mean_by(c, "Cancer_Type", "Overall_Risk_Score")["Lung"]]),

    # ---------------------------------------------------------------- netflix
    dict(id="n1", dataset="netflix", kind="group-by",
         question="How many movies and how many TV shows are there?",
         numbers=lambda n: [(n["type"] == "Movie").sum(), (n["type"] == "TV Show").sum()]),
    dict(id="n2", dataset="netflix", kind="group-by",
         question="Which value of the country column appears most often, and how many titles have it?",
         text=["United States"], numbers=lambda n: [n["country"].value_counts().iloc[0]]),
    dict(id="n3", dataset="netflix", kind="filter",
         question="How many titles were released in 2020?",
         numbers=lambda n: [(n["release_year"] == 2020).sum()]),
    dict(id="n4", dataset="netflix", kind="group-by",
         question="What is the most common rating, and how many titles have it?",
         text=["TV-MA"], numbers=lambda n: [n["rating"].value_counts().iloc[0]]),
    dict(id="n5", dataset="netflix", kind="data quality",
         question="How many titles have no director listed?",
         numbers=lambda n: [n["director"].isna().sum()]),
    dict(id="n6", dataset="netflix", kind="chart",
         question="Create a stacked area chart of Movies and TV Shows released each year.",
         chart=True),

    # ---------------------------------------------------------------- uber
    dict(id="u1", dataset="uber", kind="filter",
         question="How many bookings were completed?",
         numbers=lambda u: [(u["Booking Status"] == "Completed").sum()]),
    dict(id="u2", dataset="uber", kind="percentage",
         question="What percentage of bookings were cancelled by the driver?",
         numbers=lambda u: [(u["Booking Status"] == "Cancelled by Driver").mean() * 100]),
    dict(id="u3", dataset="uber", kind="group-by",
         question="Which vehicle type has the highest average Booking Value, and what is it?",
         text=["Go Sedan"], numbers=lambda u: [_mean_by(u, "Vehicle Type", "Booking Value").max()]),
    dict(id="u4", dataset="uber", kind="date filter",
         question="What was the total Booking Value in March 2024?",
         numbers=lambda u: [_march(u)["Booking Value"].sum()]),
    dict(id="u5", dataset="uber", kind="cleaning",
         question="Drop the rows where Booking Value is missing. How many rows are left?",
         shape=lambda u: (int(u["Booking Value"].notna().sum()), u.shape[1]),
         numbers=lambda u: [u["Booking Value"].notna().sum()]),
    dict(id="u6", dataset="uber", kind="root-cause",
         question="Why did the total Booking Value change from March to April 2024?",
         tool="explain_change",
         numbers=lambda u: [u.loc[u["Date"].str.startswith("2024-04"), "Booking Value"].sum()
                            - _march(u)["Booking Value"].sum()]),
    dict(id="u7", dataset="uber", kind="saved rule",
         question="What was the total revenue in March 2024?",
         rules=["Revenue means the sum of Booking Value for Completed rides only"],
         numbers=lambda u: [_march(u).loc[_march(u)["Booking Status"] == "Completed", "Booking Value"].sum()]),
]
