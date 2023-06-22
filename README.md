# traffic-identification
Research project on the identification of user traffic profiles in 5G (NSA and SA). 

## `ml 1.0

```log
>> sgd 0.6856
[[668 452]
 [252 867]]
>> svc 0.7226
[[953 167]
 [454 665]]
>> rf  0.8906
[[ 979  141]
 [ 104 1015]]
>> mlp 0.7678
[[815 305]
 [215 904]]
>> tree 0.7946
[[883 237]
 [223 896]]
>> xgb 0.9513
[[1052   68]
 [  41 1078]]
>> cat 0.9451
[[1044   76]
 [  47 1072]]
>> lgb 0.9460
[[1049   71]
 [  50 1069]]
```

```log
Best parameters found with logloss=0.1352 and accuracy=0.9464: 
num_leaves           3972     
max_depth            26       
learning_rate        0.4733219859672686
n_estimators         1000     
min_split_gain       0.12470872309074599
min_child_samples    26       
subsample            0.9739739662393342
subsample_freq       4        
reg_alpha            0.3152868684948474
reg_lambda           0.06595077881442596
```

```log
PDSCH basic_info    dir                        0
PDSCH short_message cr                         130
PDSCH short_message harq                       12
PDSCH short_message k1                         0
PDSCH short_message mod                        0
PDSCH short_message prb_start                  40
PDSCH short_message prb_end                    32
PDSCH short_message prb_len                    41
PDSCH short_message retx                       2
PDSCH short_message tb_len                     148
PDCCH basic_info    dir                        17
PDCCH short_message L                          7
PDCCH short_message cce_index                  539
PDCCH short_message dci                        0
PDCCH long_message  harq                       6
PDCCH long_message  mcs1                       126
PDCCH long_message  new_data_indicator1        19
PDCCH long_message  new_data_indicator2        5
PDCCH long_message  precoding_info             0
PDCCH long_message  resource_allocation_header 3
PDCCH long_message  rv_idx1                    2
PDCCH long_message  rv_idx2                    16
PDCCH long_message  tb_swap_flag               0
PDCCH long_message  tpc_command                14
PUCCH basic_info    dir                        5
PUCCH short_message ack                        6
PUCCH short_message epre                       281
PUCCH short_message format                     37
PUCCH short_message n                          167
PUCCH short_message occ                        0
PUCCH short_message prb_start                  0
PUCCH short_message prb_end                    0
PUCCH short_message prb_len                    0
PUCCH short_message prb2                       0
PUCCH short_message snr                        133
PUCCH short_message symb_start                 0
PUCCH short_message symb_end                   0
PUCCH short_message symb_len                   0
SRS   basic_info    dir                        0
SRS   short_message epre                       103
SRS   short_message prb_start                  13
SRS   short_message prb_end                    0
SRS   short_message prb_len                    0
SRS   short_message snr                        60
SRS   short_message symb_start                 0
SRS   short_message symb_end                   0
SRS   short_message symb_len                   0
SRS   short_message ta                         4
PUSCH basic_info    dir                        1
PUSCH short_message crc                        0
PUSCH short_message epre                       131
PUSCH short_message harq                       21
PUSCH short_message mod                        0
PUSCH short_message prb_start                  14
PUSCH short_message prb_end                    8
PUSCH short_message prb_len                    24
PUSCH short_message retx                       4
PUSCH short_message rv_idx                     0
PUSCH short_message snr                        99
PUSCH short_message symb_start                 0
PUSCH short_message symb_end                   1
PUSCH short_message symb_len                   0
PUSCH short_message ta                         73
PUSCH short_message tb_len                     28
PHICH basic_info    dir                        15
PHICH short_message group                      36
PHICH short_message hi                         15
PHICH short_message seq                        17
```