# Evaluation results

22 questions on 3 sample datasets; expected answers computed with pandas.

## Overall

| model              | critic   |   questions | accuracy   |   avg_time_s |   avg_steps |   tool_errors_per_q |   critic_flagged |
|:-------------------|:---------|------------:|:-----------|-------------:|------------:|--------------------:|-----------------:|
| openai/gpt-oss-20b | on       |          22 | 100.0%     |        21.15 |        1.27 |                   0 |                0 |

## Accuracy by question type (%)

| kind              |   ('openai/gpt-oss-20b', 'on') |
|:------------------|-------------------------------:|
| aggregate         |                            100 |
| chart             |                            100 |
| cleaning          |                            100 |
| data prep + chart |                            100 |
| data quality      |                            100 |
| date filter       |                            100 |
| filter            |                            100 |
| filter+aggregate  |                            100 |
| group-by          |                            100 |
| lookup            |                            100 |
| percentage        |                            100 |
| root-cause        |                            100 |
| saved rule        |                            100 |

## Failed questions

None.
