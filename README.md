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

## enb1009 ml
```log
Training of model sgd over 284367 records completed in 21.553266763687134 seconds
Testing of model sgd over 71092 records completed in 0.047868967056274414 seconds
>> sgd 0.7873 0.7787
[[6185   62    0    3  686   11   81   23  183]
 [  45 6590    0    1   33    6   19  516  103]
 [   0    1 7673    0    0   24   12    0   66]
 [  11    3   11 6256    1   61  255    6 1017]
 [1900   89    1    5 3116   10   16 1650  181]
 [ 116   22   36   21    4 2603  492    2  756]
 [  44   16    6   77    5  364 8837   15  914]
 [  65  167    3   47  187   24    2 7871  127]
 [  63   42   62 2810    7  510 1003   23 6837]]
Training of model rf over 284367 records completed in 97.920086145401 seconds
Testing of model rf over 71092 records completed in 1.8825230598449707 seconds
>> rf  0.8887 0.8876
[[6745    7    0    0  448    5    9    5   15]
 [  22 7158    0    1    8    1    6  104   13]
 [   0    0 7692    1    0   14    5    0   64]
 [   6    1    1 6680    2   52   78    9  792]
 [ 965   11    0    0 5393    3    5  579   12]
 [  16    6    0   12    1 3337  277    0  403]
 [   7    3    0    5    3  248 9748    2  262]
 [  55  182    0    8  409    9    9 7776   45]
 [  29    6    6 1823   12  251  572    6 8652]]
Training of model tree over 284367 records completed in 1.4771840572357178 seconds
Testing of model tree over 71092 records completed in 0.0748891830444336 seconds
>> tree 0.7610 0.7594
[[5211  228    0   22 1312   79  186   81  115]
 [ 242 6224    0   40  137   31   90  474   75]
 [   0    0 7686    7    0   19    9    1   54]
 [  30   34   10 5134   32   74  120   51 2136]
 [1258  145    0   38 4293   20   64 1073   77]
 [  51   32   19   69   19 2929  417   16  500]
 [ 185   86    6  145   76  475 8403   46  856]
 [  75  468    1   46 1141   23   41 6633   65]
 [  99   89   49 2119   85  533  696  100 7587]]
Training of model xgb over 284367 records completed in 244.75306701660156 seconds
Testing of model xgb over 71092 records completed in 0.21031427383422852 seconds
>> xgb 0.9067 0.9060
[[6753   17    0    1  415    5    2    4   37]
 [  16 7185    0    0    1    2    0   90   19]
 [   1    0 7690    1    0    7    2    0   75]
 [   9    0    0 6676    1   19   46    0  870]
 [ 633   21    0    0 5744    1    2  523   44]
 [  22    1    0    9    3 3344  162    0  511]
 [   7    1    0    6    1  222 9686    0  355]
 [  45   83    0    5  303    8    0 7993   56]
 [  31    0    5 1423    8  192  305    5 9388]]
Training of model cat over 284367 records completed in 214.9320366382599 seconds
Testing of model cat over 71092 records completed in 3.4867300987243652 seconds
>> cat 0.9031 0.9013
[[6777   21    0    0  406    4    1    3   22]
 [  21 7188    0    0    1    0    0   88   15]
 [   1    0 7683    1    0   17    4    0   70]
 [   3    2    0 6706    4   28   53    4  821]
 [ 866   22    0    0 5591    1    0  467   21]
 [  18    2    5    5    4 3306  196    1  515]
 [   4    2    0    7    3  261 9605    0  396]
 [  62  106    0    0  265    6    0 8022   32]
 [  21    2   17 1381    9  240  358    2 9327]]
Training of model lgb over 284367 records completed in 13.089590549468994 seconds
Testing of model lgb over 71092 records completed in 0.22523832321166992 seconds
>> lgb 0.9143 0.9136
[[6754    9    1    1  432    4    1    4   28]
 [   8 7251    0    0    1    0    0   42   11]
 [   1    0 7694    1    0    9    3    0   68]
 [   6    0    1 6698    5   15   41    1  854]
 [ 569    3    1    0 5948    2    2  425   18]
 [  10    0    0    8    9 3393  196    1  435]
 [   6    1    0    4    1  224 9770    0  272]
 [  39   75    0    5  288    3    0 8045   38]
 [  27    0    7 1370   14  190  299    4 9446]]
```