# traffic-identification
Research project on the identification of user traffic profiles in 5G (NSA and SA). 

## `ml 1.0`
```log
>> sgd 0.6478
[[1020 1133]
 [ 404 1807]]
>> svc 0.6881
[[1684  469]
 [ 892 1319]]
>> rf  0.8875
[[1883  270]
 [ 221 1990]]
>> mlp 0.7438
[[1421  732]
 [ 386 1825]]
>> tree 0.7727
[[1668  485]
 [ 507 1704]]
>> xgb 0.9283
[[1973  180]
 [ 133 2078]]
>> cat 0.9251
[[1949  204]
 [ 123 2088]]
>> lgb 0.9253
[[1961  192]
 [ 134 2077]]
```

```log
Best parameters found with logloss=0.1718 and accuracy=0.9267: 
num_leaves           2743     
max_depth            24       
learning_rate        0.4394407332890032
n_estimators         50       
min_split_gain       0.16141325462062206
min_child_samples    10       
subsample            0.4054846083040434
subsample_freq       0        
reg_alpha            0.6784963878829607
reg_lambda           0.37660323302045556
```

```log
Evaluating LightGBM model feature importance...
PDSCH basic_info    dir                        1
PDSCH short_message cr                         194
PDSCH short_message harq                       7
PDSCH short_message k1                         0
PDSCH short_message mod                        3
PDSCH short_message prb_start                  44
PDSCH short_message prb_end                    24
PDSCH short_message prb_len                    26
PDSCH short_message retx                       10
PDSCH short_message tb_len                     169
PDCCH basic_info    dir                        19
PDCCH short_message L                          18
PDCCH short_message cce_index                  660
PDCCH short_message dci                        0
PDCCH long_message  harq                       1
PDCCH long_message  mcs1                       130
PDCCH long_message  new_data_indicator1        16
PDCCH long_message  new_data_indicator2        14
PDCCH long_message  precoding_info             0
PDCCH long_message  resource_allocation_header 7
PDCCH long_message  rv_idx1                    5
PDCCH long_message  rv_idx2                    15
PDCCH long_message  tb_swap_flag               0
PDCCH long_message  tpc_command                38
PUCCH basic_info    dir                        20
PUCCH short_message ack                        23
PUCCH short_message epre                       578
PUCCH short_message format                     40
PUCCH short_message n                          178
PUCCH short_message occ                        0
PUCCH short_message prb_start                  0
PUCCH short_message prb_end                    0
PUCCH short_message prb_len                    0
PUCCH short_message prb2                       0
PUCCH short_message snr                        293
PUCCH short_message symb_start                 0
PUCCH short_message symb_end                   0
PUCCH short_message symb_len                   0
SRS   basic_info    dir                        2
SRS   short_message epre                       157
SRS   short_message prb_start                  21
SRS   short_message prb_end                    0
SRS   short_message prb_len                    1
SRS   short_message snr                        129
SRS   short_message symb_start                 0
SRS   short_message symb_end                   0
SRS   short_message symb_len                   0
SRS   short_message ta                         14
PUSCH basic_info    dir                        5
PUSCH short_message crc                        0
PUSCH short_message epre                       95
PUSCH short_message harq                       18
PUSCH short_message mod                        0
PUSCH short_message prb_start                  21
PUSCH short_message prb_end                    10
PUSCH short_message prb_len                    12
PUSCH short_message retx                       5
PUSCH short_message rv_idx                     0
PUSCH short_message snr                        83
PUSCH short_message symb_start                 0
PUSCH short_message symb_end                   0
PUSCH short_message symb_len                   0
PUSCH short_message ta                         90
PUSCH short_message tb_len                     32
PHICH basic_info    dir                        16
PHICH short_message group                      59
PHICH short_message hi                         12
PHICH short_message seq                        15
```

