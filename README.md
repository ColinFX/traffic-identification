# traffic-identification
Research project on the identification of user traffic profiles in 5G (NSA and SA). 

## nr 1st-example ml 1.0

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

## enb0926 ml
```log
Loading data...
Comparing ML models...
Training of model sgd over 340220 records completed in 8.6504545211792 seconds
Testing of model sgd over 85056 records completed in 0.0482935905456543 seconds
>> sgd 0.9833 0.8076
[[21611    42     0     0    24     1]
 [  159  6772     2     3   143    10]
 [    9     1   319     0   297     0]
 [   16     0     1    85   463     4]
 [  107    17    29     7 24745     0]
 [    1    62     0     0    25 30101]]
Training of model rf over 340220 records completed in 92.66327047348022 seconds
Testing of model rf over 85056 records completed in 1.9781503677368164 seconds
>> rf  0.9959 0.9601
[[21660    10     0     0     8     0]
 [   27  7022     0     1    27    12]
 [    1     2   549    15    59     0]
 [    0     4    28   460    76     1]
 [   19    13     7    15 24851     0]
 [    1    20     0     0     5 30163]]
Training of model tree over 340220 records completed in 1.827728271484375 seconds
Testing of model tree over 85056 records completed in 0.11272287368774414 seconds
>> tree 0.9713 0.8600
[[21278   155     5     8   226     6]
 [  173  6521     6     8   219   162]
 [    1     3   452    76    92     2]
 [    9     8    94   347   107     4]
 [  256   220    97   149 24123    60]
 [   10   210     4     7    63 29895]]
Training of model xgb over 340220 records completed in 196.9267077445984 seconds
Testing of model xgb over 85056 records completed in 0.206712007522583 seconds
>> xgb 0.9973 0.9703
[[21667     8     0     0     3     0]
 [    5  7077     1     0     3     3]
 [    2     1   566    11    46     0]
 [    0     3    16   477    73     0]
 [    9    13     4     7 24872     0]
 [    0    20     0     0     0 30169]]
Training of model cat over 340220 records completed in 166.54155850410461 seconds
Testing of model cat over 85056 records completed in 5.637664794921875 seconds
>> cat 0.9961 0.9629
[[21667     6     0     0     5     0]
 [   19  7049     1     0    17     3]
 [    1     1   558     9    57     0]
 [    3     4     8   445   109     0]
 [   15    19     6    11 24854     0]
 [    0    38     0     0     0 30151]]
Training of model lgb over 340220 records completed in 12.20529294013977 seconds
Testing of model lgb over 85056 records completed in 0.18735814094543457 seconds
>> lgb 0.9768 0.8919
[[21547    42     2     3    72    12]
 [  122  6561     8   110    49   239]
 [    2     2   547    15    60     0]
 [    7    24    29   403    84    22]
 [  235    67    47    78 24432    46]
 [  192   122     7   185    87 29596]]
```