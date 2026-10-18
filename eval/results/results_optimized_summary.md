# Evaluation results

21 questions on 3 sample datasets; expected answers computed with pandas.

## Overall

| model              | critic   |   questions | accuracy   |   avg_time_s |   avg_steps |   tool_errors_per_q |   critic_flagged |
|:-------------------|:---------|------------:|:-----------|-------------:|------------:|--------------------:|-----------------:|
| openai/gpt-oss-20b | on       |          21 | 95.2%      |        22.52 |        1.19 |                   0 |                1 |

## Accuracy by question type (%)

| kind             |   ('openai/gpt-oss-20b', 'on') |
|:-----------------|-------------------------------:|
| aggregate        |                            100 |
| chart            |                            100 |
| cleaning         |                              0 |
| data quality     |                            100 |
| date filter      |                            100 |
| filter           |                            100 |
| filter+aggregate |                            100 |
| group-by         |                            100 |
| lookup           |                            100 |
| percentage       |                            100 |
| root-cause       |                            100 |
| saved rule       |                            100 |

## Failed questions

| model              | critic   | qid   |   expected | answer                                                                                     |
|:-------------------|:---------|:------|-----------:|:-------------------------------------------------------------------------------------------|
| openai/gpt-oss-20b | on       | u5    |    102,000 | After removing all rows with missing Booking Value, the dataset contains ==102,000== rows. |
