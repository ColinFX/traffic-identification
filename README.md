
# traffic-identification
Research project on the identification of user traffic profiles in 5G (NSA and SA). 

## ml 0.1
```log
>> sgd 0.6588
[[901 495]
 [460 943]]

>> svc 0.7213
[[1189  207]
 [ 573  830]]

>> rf  0.8903
[[1221  175]
 [ 132 1271]]

>> mlp 0.7531
[[ 933  463]
 [ 228 1175]]

>> tree 0.8003
[[1104  292]
 [ 267 1136]]

>> xgb 0.9521
[[1307   89]
 [  45 1358]]

>> lgb 0.9471
[[1305   91]
 [  57 1346]]
```

## ml 0.2
```log
Best parameters found with logloss=0.1310 and accuracy=0.9486: 
num_leaves           339      
max_depth            18       
learning_rate        0.4919025266695265
n_estimators         200      
min_split_gain       0.1927155954273386
min_child_samples    10       
subsample            0.8799518725707276
subsample_freq       0        
reg_alpha            0.07597720488208505
reg_lambda           0.8335676629734721
Evaluating LightGBM model feature importance...
PDSCH basic_info    dir                        3
PDSCH short_message cr                         70
PDSCH short_message harq                       11
PDSCH short_message k1                         0
PDSCH short_message mod                        2
PDSCH short_message prb_start                  18
PDSCH short_message prb_end                    8
PDSCH short_message prb_len                    17
PDSCH short_message retx                       6
PDSCH short_message tb_len                     56
PDCCH basic_info    dir                        10
PDCCH short_message L                          16
PDCCH short_message cce_index                  268
PDCCH short_message dci                        0
PDCCH long_message  harq                       0
PDCCH long_message  mcs1                       42
PDCCH long_message  new_data_indicator1        6
PDCCH long_message  new_data_indicator2        6
PDCCH long_message  precoding_info             0
PDCCH long_message  resource_allocation_header 1
PDCCH long_message  rv_idx1                    6
PDCCH long_message  rv_idx2                    13
PDCCH long_message  tb_swap_flag               0
PDCCH long_message  tpc_command                16
PUCCH basic_info    dir                        0
PUCCH short_message ack                        2
PUCCH short_message epre                       186
PUCCH short_message format                     27
PUCCH short_message n                          102
PUCCH short_message occ                        0
PUCCH short_message prb_start                  0
PUCCH short_message prb_end                    0
PUCCH short_message prb_len                    0
PUCCH short_message prb2                       0
PUCCH short_message snr                        65
PUCCH short_message symb_start                 0
PUCCH short_message symb_end                   0
PUCCH short_message symb_len                   0
SRS   basic_info    dir                        0
SRS   short_message epre                       63
SRS   short_message prb_start                  7
SRS   short_message prb_end                    0
SRS   short_message prb_len                    0
SRS   short_message snr                        36
SRS   short_message symb_start                 0
SRS   short_message symb_end                   0
SRS   short_message symb_len                   0
SRS   short_message ta                         2
PUSCH basic_info    dir                        4
PUSCH short_message crc                        0
PUSCH short_message epre                       15
PUSCH short_message harq                       6
PUSCH short_message mod                        0
PUSCH short_message prb_start                  1
PUSCH short_message prb_end                    0
PUSCH short_message prb_len                    9
PUSCH short_message retx                       9
PUSCH short_message rv_idx                     0
PUSCH short_message snr                        18
PUSCH short_message symb_start                 0
PUSCH short_message symb_end                   0
PUSCH short_message symb_len                   0
PUSCH short_message ta                         17
PUSCH short_message tb_len                     8
PHICH basic_info    dir                        9
PHICH short_message group                      24
PHICH short_message hi                         12
PHICH short_message seq                        12
```