Mean value of lengths: 11.879681587219238\
Proportion of lengths in range 0-5: 0.04422394532790117\
Proportion of lengths in range 5-10: 0.20370613746878696\
Proportion of lengths in range 10-15: 0.3641740044683927\
Proportion of lengths in range 15-20: 0.3878959127349192

set the max_length = 15
| Model               | Test Set | Short Sequence | Test Entropy | Short Seq Entropy |
|---------------------|----------|----------------|--------------|--------------------|
| lstm_attention with pretrain     | 42.70%   |  43.32%        |0.7957        |0.7897              |
| lstm_attention without pretrain      | 40.56%   |  41.32%        |0.8298        |0.8241              |
| Decoder only        | 56.04% | 56.52% |    0.6612          |      0.6591              |
| lstm with pretrain  |  44.02%  | 43.84%         |0.7808        | 0.7822             |
| lstm without pretrain | 44.65% | 44.48%         |0.7809        | 0.7845             |
| baseline            | 35.56%   | 35.52%         | 0.8755       | 0.8811             |
