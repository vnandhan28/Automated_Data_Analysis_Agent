# Evaluation results

21 questions on 3 sample datasets; expected answers computed with pandas.

## Overall

| model               | critic   |   questions | accuracy   |   avg_time_s |   avg_steps |   tool_errors_per_q |   critic_flagged |
|:--------------------|:---------|------------:|:-----------|-------------:|------------:|--------------------:|-----------------:|
| openai/gpt-oss-120b | off      |          21 | 100.0%     |        24.96 |        2.14 |                0.05 |                0 |
| openai/gpt-oss-120b | on       |           7 | 100.0%     |        26.89 |        2    |                0.14 |                0 |
| openai/gpt-oss-20b  | off      |          18 | 100.0%     |        34.32 |        2.11 |                0    |                0 |
| openai/gpt-oss-20b  | on       |           9 | 100.0%     |        41.87 |        2    |                0    |                0 |

## Accuracy by question type (%)

| kind             |   ('openai/gpt-oss-120b', 'off') |   ('openai/gpt-oss-120b', 'on') |   ('openai/gpt-oss-20b', 'off') |   ('openai/gpt-oss-20b', 'on') |
|:-----------------|---------------------------------:|--------------------------------:|--------------------------------:|-------------------------------:|
| aggregate        |                              100 |                             100 |                             nan |                            100 |
| chart            |                              100 |                             100 |                             100 |                            100 |
| cleaning         |                              100 |                             nan |                             100 |                            nan |
| data quality     |                              100 |                             nan |                             100 |                            nan |
| date filter      |                              100 |                             nan |                             100 |                            nan |
| filter           |                              100 |                             100 |                             100 |                            100 |
| filter+aggregate |                              100 |                             100 |                             100 |                            100 |
| group-by         |                              100 |                             100 |                             100 |                            100 |
| lookup           |                              100 |                             100 |                             100 |                            100 |
| percentage       |                              100 |                             nan |                             100 |                            nan |
| root-cause       |                              100 |                             nan |                             100 |                            100 |
| saved rule       |                              100 |                             nan |                             100 |                            nan |

## Failed questions

None.

_4 API errors (rate limits etc.) were excluded and will be retried._
