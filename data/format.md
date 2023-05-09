# Log File Format

## [PHY]

### Example

```log
09:50:00.317 [PHY] DL 0009 04 4602   233.2 PDSCH: harq=2 prb=46:5 symb=1:13 k1=7 nl=2 CW0: tb_len=912 mod=6 rv_idx=0 cr=0.85 retx=0

09:50:00.318 [PHY] DL 0009 04 4602   233.3 PDCCH: ss_id=2 cce_index=0 al=2 dci=1_1 
    rb_alloc=0x32 
    time_domain_rsc=0 
    mcs1=19 
    ndi1=1 
    rv_idx1=0 
    harq_process=3 
    dai=1 
    tpc_command=1 
    pucch_rsc=0 
    harq_feedback_timing=2 
    antenna_ports=2 
    srs_request=0 
    dmrs_seq_init=0
```

### Format

```log
time layer dir ue_id cell_id rnti sfn channel: short_message
    long_message
```

### Parameters

`ss_id` stands for **search space ID**. It is used to identify the search space in which the PDCCH (Physical Downlink Control Channel) is transmitted¹. The search space is a set of resource blocks (RBs) that are used to transmit PDCCH². 

`cce_index` stands for **Control Channel Element ID**. It is used to identify the CCE (Control Channel Element) in which the PDCCH is transmitted². CCE is a group of resource elements (REs) that are used to transmit PDCCH². 

`dci` stands for **Downlink Control Information**. It is used to carry downlink control information on the PDCCH. 

`mcs` stands for **Modulation and Coding Scheme**. It is used to define the modulation and coding scheme used for downlink data transmission. The higher the MCS value, the higher the modulation order and the more bits per symbol are transmitted. This means that higher MCS values can transmit more data per unit time but require a better signal-to-noise ratio (SNR) to maintain a certain level of reliability.

## [GTPU]

### Example

```log
09:50:00.317 [GTPU] FROM 127.0.1.100:2152 G-PDU TEID=0x52cff9de SDU_len=216: IP/TCP 216.58.213.74:443 > 192.168.3.2:50596
        0000:  30 ff 00 94 52 cf f9 de  45 80 00 94 a6 5c 00 00  0...R...E....\..
        0010:  77 06 2b 58 d8 3a d5 4a  c0 a8 03 02 01 bb c5 a4  w.+X.:.J........
        ...
```

### Format

```log 
time layer dir ip_address short_message
    long message
```

### Parameters

`G-PDU` stands for **GPRS Protocol Data Unit**. It is a message that carries the original data packet from a user or external PDN equipment.

GPRS Tunneling Protocol (GTP) is a group of IP-based communications protocols used to carry general packet radio service (GPRS) within GSM, UMTS, LTE and 5G NR radio networks. GTP and Proxy Mobile IPv6 based interfaces are specified on various interface points in 3GPP architectures.

`TEID` stands for **Tunnel Endpoint Identifier**. The GPRS tunneling protocol (GTP) stack assigns a unique TEID to each GTP user connection (bearer) to the peers. The TEID is a 32-bit number field in the GTP (GTP-C or GTP-U) packet.

The TEID in the GTP-U header is used to de-multiplex traffic incoming from remote tunnel endpoints so that it is delivered to the User plane entities in a way that allows multiplexing of different users, different packet protocols and different QoS levels.

`SDU_len` stands for **Service Data Unit Length**. In Open Systems Interconnection (OSI) terminology, a service data unit (SDU) is a unit of data that has been passed down from an OSI layer or sublayer to a lower layer. This unit of data (SDU) has not yet been encapsulated into a protocol data unit (PDU) by the lower layer.

## [Others]

```log
PDCP    time layer dir ue_id    bearer              message
RRC     time layer dir ue_id    cell_id channel:    str
NAS     time layer dir ue_id    protocol:           str
```