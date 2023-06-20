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
Best parameters found with logloss=0.1674 and accuracy=0.9276: 
num_leaves           3008     
max_depth            26       
learning_rate        0.44856010254318723
n_estimators         100      
min_split_gain       0.17613702828010502
min_child_samples    10       
subsample            0.9650050826624225
subsample_freq       2        
reg_alpha            0.3328823536919523
reg_lambda           0.12385012284783442
```

