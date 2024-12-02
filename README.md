# The action prediction of Sepsis patient 
There are two actions for sepsis patient: 



Mean value of lengths: 11.879681587219238\
Proportion of lengths in range 0-5: 0.04422394532790117\
Proportion of lengths in range 5-10: 0.20370613746878696\
Proportion of lengths in range 10-15: 0.3641740044683927\
Proportion of lengths in range 15-20: 0.3878959127349192

## condifent rate
set the max_length = 15
| Model               | Test Set | Short Sequence | Test Entropy | Short Seq Entropy |
|---------------------|----------|----------------|--------------|--------------------|
| lstm_attention with pretrain     | 42.70%   |  43.32%        |0.7957        |0.7897              |
| lstm_attention without pretrain      | 40.56%   |  41.32%        |0.8298        |0.8241              |
| Decoder only        | 56.04% | 56.52% |    0.6612          |      0.6591              |
| lstm with pretrain  |  44.02%  | 43.84%         |0.7808        | 0.7822             |
| lstm without pretrain | 44.65% | 44.48%         |0.7809        | 0.7845             |
| baseline            | 35.56%   | 35.52%         | 0.8755       | 0.8811             |

set the max_length = 20
| Model               | Test Set | Short Sequence | Test Entropy | Short Seq Entropy |
|---------------------|----------|----------------|--------------|--------------------|
| Decoder only        | 50.12% | 48.92% |    0.7333          |      0.7403              |
| baseline            | 33.54%   | 33.76%        | 0.9006       | 0.9053             |


## Accuracy on test set

- Class 0: count = 42078.0, weight = 0.6613 
- Class 1: count = 25614.0, weight = 1.0863
- Class 2: count = 24865.0, weight = 1.1190
- Class 3: count = 25297.0, weight = 1.0999
- Class 4: count = 26923.0, weight = 1.0335

seq_length = 15
| Model               | Class 0 | Class 1 |Class 2| Class 3| Class 4| Overall |
|---------------------|----------|----------------|--------------|--------------------|--------------------|--------------------|
| Decoder only        |  60.76% (5995/9866) AUC:0.7974  |22.84% (1443/6317) AUC: 0.6203  |  23.18% (1461/6303) AUC: 0.6263|    25.11% (1634/6508) AUC: 0.6234 | 47.13% (3306/7014) AUC: 0.7637| 38.43% AUC: 0.6862|
| lstm with pretrain  |  59.90% (5910/9866) AUC: 0.8050 | 25.09% (1585/6317) AUC: 0.6376        |25.72% (1621/6303)  AUC: 0.6372 | 27.12% (1765/6508) AUC: 0.6374|49.76% (3490/7014) AUC: 0.7889| 39.91% AUC: 0.70122|
| baseline            |  59.05% (5826/9866)  AUC: 0.8110   | 25.65% (1620/6317) AUC: 0.6290 | 25.67% (1618/6303) AUC: 0.6358     | 28.21% (1836/6508) AUC: 0.6436 |49.64% (3482/7014) AUC: 0.7886| 39.94% AUC: 0.7016|

Performance on **short sequence** (seq_length = 5)

| Model               | Class 0 | Class 1 |Class 2| Class 3| Class 4|overall|
|---------------------|----------|----------------|--------------|--------------------|--------------------|--------------------|
| Decoder only        |  63.36% (434/685)  |22.22% (96/432)  |   20.31% (91/448)|    23.81% (100/420)           | 46.80% (241/515)|38.48% (962/2500)|
| lstm with pretrain  |  61.75% (423/685)  | 28.24% (122/432)         |23.21% (104/448)        | 29.05% (122/420)|52.23% (269/515)| 41.60% (1040/2500) |
| baseline            | 62.34% (427/685)| 25.93% (112/432)         | 22.77% (102/448)       | 30.24% (127/420)             |51.84% (267/515)|41.40% (1035/2500) |


- Class 0: count = 42078.0, weight = 0.0432
- Class 1: count = 25614.0, weight = 1.5691
- Class 2: count = 24865.0, weight = 1.6068
- Class 3: count = 25297.0, weight = 1.5848
- Class 4: count = 26923.0, weight = 0.1960

| Model               | Class 0 | Class 1 |Class 2| Class 3| Class 4| Overall |
|---------------------|----------|----------------|--------------|--------------------|--------------------|--------------------|
| Decoder only        |  51.68% (5099/9866) AUC:0.7974  |29.41% (1858/6317) AUC: 0.6203  |  27.03% (1704/6303) AUC: 0.6263|    29.55% (1923/6508) AUC: 0.6234 | 39.86% (2796/7014) AUC: 0.7637| 37.16% AUC: 0.6817|
| lstm with pretrain  |  47.99% (4735/9866) AUC: 0.7982 | 32.66% (2063/6317) AUC: 0.6376        |30.29% (1909/6303)  AUC: 0.6372 | 33.39% (2173/6508) AUC: 0.6374|36.83% (2583/7014) AUC: 0.7889| 37.39% AUC: 0.6936|
| baseline            |  59.05% (5826/9866)  AUC: 0.8110   | 25.65% (1620/6317) AUC: 0.6290 | 25.67% (1618/6303) AUC: 0.6358     | 28.21% (1836/6508) AUC: 0.6436 |49.64% (3482/7014) AUC: 0.7886| 39.94% AUC: 0.7016|

