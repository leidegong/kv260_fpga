   0  CFG   PAGE = 8192
   1  CFG   HEAD_DIM = 128
   2  CFG   N_Q = 16
   3  CFG   N_KV = 8
   4  CFG   CTX_MAX = 4096
   5  CFG   KV_BITS = 8
   6  CFG   KV_DATA = 524288
   7  CFG   KV_SCALE = 8192
   8  CFG   EPS = 897988541
   9  CFG   ROPE_THETA = 1232348160
  10  CFG   SCRATCH = 233728
  11  CFG   GROUP = 128
  12  CFG   BATCH = 8
  13  EMB   flags=00 dst=0 src0=0 src1=0 n=2048 aux=16 lm_head
  14  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L0.norms
  15  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
  16  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L0.q_proj
  17  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L0.k_proj
  18  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L0.v_proj
  19  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
  20  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
  21  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
  22  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
  23  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=0 L0.K0
  24  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=0 L0.K0
  25  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L0.o_proj
  26  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
  27  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L0.gate_proj
  28  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L0.up_proj
  29  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
  30  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L0.down_proj
  31  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L1.norms
  32  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
  33  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L1.q_proj
  34  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L1.k_proj
  35  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L1.v_proj
  36  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
  37  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
  38  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
  39  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
  40  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=1 L1.K0
  41  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=1 L1.K0
  42  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L1.o_proj
  43  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
  44  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L1.gate_proj
  45  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L1.up_proj
  46  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
  47  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L1.down_proj
  48  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L2.norms
  49  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
  50  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L2.q_proj
  51  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L2.k_proj
  52  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L2.v_proj
  53  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
  54  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
  55  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
  56  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
  57  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=2 L2.K0
  58  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=2 L2.K0
  59  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L2.o_proj
  60  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
  61  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L2.gate_proj
  62  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L2.up_proj
  63  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
  64  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L2.down_proj
  65  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L3.norms
  66  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
  67  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L3.q_proj
  68  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L3.k_proj
  69  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L3.v_proj
  70  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
  71  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
  72  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
  73  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
  74  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=3 L3.K0
  75  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=3 L3.K0
  76  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L3.o_proj
  77  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
  78  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L3.gate_proj
  79  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L3.up_proj
  80  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
  81  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L3.down_proj
  82  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L4.norms
  83  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
  84  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L4.q_proj
  85  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L4.k_proj
  86  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L4.v_proj
  87  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
  88  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
  89  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
  90  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
  91  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=4 L4.K0
  92  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=4 L4.K0
  93  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L4.o_proj
  94  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
  95  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L4.gate_proj
  96  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L4.up_proj
  97  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
  98  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L4.down_proj
  99  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L5.norms
 100  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 101  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L5.q_proj
 102  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L5.k_proj
 103  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L5.v_proj
 104  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 105  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 106  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 107  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 108  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=5 L5.K0
 109  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=5 L5.K0
 110  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L5.o_proj
 111  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 112  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L5.gate_proj
 113  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L5.up_proj
 114  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 115  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L5.down_proj
 116  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L6.norms
 117  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 118  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L6.q_proj
 119  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L6.k_proj
 120  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L6.v_proj
 121  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 122  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 123  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 124  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 125  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=6 L6.K0
 126  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=6 L6.K0
 127  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L6.o_proj
 128  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 129  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L6.gate_proj
 130  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L6.up_proj
 131  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 132  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L6.down_proj
 133  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L7.norms
 134  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 135  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L7.q_proj
 136  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L7.k_proj
 137  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L7.v_proj
 138  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 139  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 140  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 141  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 142  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=7 L7.K0
 143  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=7 L7.K0
 144  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L7.o_proj
 145  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 146  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L7.gate_proj
 147  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L7.up_proj
 148  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 149  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L7.down_proj
 150  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L8.norms
 151  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 152  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L8.q_proj
 153  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L8.k_proj
 154  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L8.v_proj
 155  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 156  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 157  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 158  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 159  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=8 L8.K0
 160  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=8 L8.K0
 161  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L8.o_proj
 162  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 163  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L8.gate_proj
 164  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L8.up_proj
 165  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 166  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L8.down_proj
 167  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L9.norms
 168  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 169  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L9.q_proj
 170  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L9.k_proj
 171  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L9.v_proj
 172  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 173  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 174  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 175  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 176  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=9 L9.K0
 177  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=9 L9.K0
 178  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L9.o_proj
 179  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 180  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L9.gate_proj
 181  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L9.up_proj
 182  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 183  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L9.down_proj
 184  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L10.norms
 185  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 186  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L10.q_proj
 187  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L10.k_proj
 188  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L10.v_proj
 189  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 190  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 191  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 192  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 193  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=10 L10.K0
 194  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=10 L10.K0
 195  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L10.o_proj
 196  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 197  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L10.gate_proj
 198  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L10.up_proj
 199  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 200  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L10.down_proj
 201  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L11.norms
 202  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 203  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L11.q_proj
 204  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L11.k_proj
 205  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L11.v_proj
 206  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 207  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 208  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 209  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 210  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=11 L11.K0
 211  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=11 L11.K0
 212  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L11.o_proj
 213  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 214  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L11.gate_proj
 215  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L11.up_proj
 216  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 217  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L11.down_proj
 218  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L12.norms
 219  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 220  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L12.q_proj
 221  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L12.k_proj
 222  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L12.v_proj
 223  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 224  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 225  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 226  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 227  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=12 L12.K0
 228  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=12 L12.K0
 229  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L12.o_proj
 230  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 231  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L12.gate_proj
 232  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L12.up_proj
 233  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 234  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L12.down_proj
 235  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L13.norms
 236  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 237  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L13.q_proj
 238  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L13.k_proj
 239  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L13.v_proj
 240  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 241  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 242  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 243  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 244  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=13 L13.K0
 245  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=13 L13.K0
 246  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L13.o_proj
 247  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 248  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L13.gate_proj
 249  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L13.up_proj
 250  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 251  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L13.down_proj
 252  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L14.norms
 253  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 254  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L14.q_proj
 255  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L14.k_proj
 256  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L14.v_proj
 257  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 258  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 259  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 260  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 261  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=14 L14.K0
 262  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=14 L14.K0
 263  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L14.o_proj
 264  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 265  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L14.gate_proj
 266  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L14.up_proj
 267  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 268  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L14.down_proj
 269  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L15.norms
 270  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 271  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L15.q_proj
 272  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L15.k_proj
 273  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L15.v_proj
 274  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 275  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 276  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 277  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 278  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=15 L15.K0
 279  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=15 L15.K0
 280  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L15.o_proj
 281  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 282  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L15.gate_proj
 283  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L15.up_proj
 284  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 285  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L15.down_proj
 286  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L16.norms
 287  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 288  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L16.q_proj
 289  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L16.k_proj
 290  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L16.v_proj
 291  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 292  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 293  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 294  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 295  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=16 L16.K0
 296  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=16 L16.K0
 297  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L16.o_proj
 298  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 299  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L16.gate_proj
 300  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L16.up_proj
 301  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 302  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L16.down_proj
 303  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L17.norms
 304  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 305  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L17.q_proj
 306  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L17.k_proj
 307  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L17.v_proj
 308  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 309  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 310  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 311  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 312  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=17 L17.K0
 313  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=17 L17.K0
 314  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L17.o_proj
 315  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 316  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L17.gate_proj
 317  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L17.up_proj
 318  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 319  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L17.down_proj
 320  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L18.norms
 321  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 322  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L18.q_proj
 323  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L18.k_proj
 324  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L18.v_proj
 325  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 326  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 327  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 328  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 329  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=18 L18.K0
 330  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=18 L18.K0
 331  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L18.o_proj
 332  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 333  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L18.gate_proj
 334  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L18.up_proj
 335  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 336  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L18.down_proj
 337  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L19.norms
 338  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 339  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L19.q_proj
 340  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L19.k_proj
 341  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L19.v_proj
 342  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 343  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 344  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 345  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 346  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=19 L19.K0
 347  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=19 L19.K0
 348  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L19.o_proj
 349  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 350  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L19.gate_proj
 351  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L19.up_proj
 352  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 353  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L19.down_proj
 354  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L20.norms
 355  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 356  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L20.q_proj
 357  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L20.k_proj
 358  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L20.v_proj
 359  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 360  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 361  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 362  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 363  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=20 L20.K0
 364  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=20 L20.K0
 365  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L20.o_proj
 366  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 367  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L20.gate_proj
 368  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L20.up_proj
 369  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 370  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L20.down_proj
 371  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L21.norms
 372  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 373  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L21.q_proj
 374  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L21.k_proj
 375  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L21.v_proj
 376  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 377  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 378  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 379  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 380  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=21 L21.K0
 381  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=21 L21.K0
 382  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L21.o_proj
 383  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 384  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L21.gate_proj
 385  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L21.up_proj
 386  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 387  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L21.down_proj
 388  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L22.norms
 389  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 390  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L22.q_proj
 391  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L22.k_proj
 392  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L22.v_proj
 393  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 394  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 395  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 396  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 397  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=22 L22.K0
 398  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=22 L22.K0
 399  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L22.o_proj
 400  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 401  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L22.gate_proj
 402  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L22.up_proj
 403  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 404  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L22.down_proj
 405  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L23.norms
 406  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 407  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L23.q_proj
 408  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L23.k_proj
 409  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L23.v_proj
 410  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 411  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 412  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 413  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 414  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=23 L23.K0
 415  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=23 L23.K0
 416  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L23.o_proj
 417  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 418  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L23.gate_proj
 419  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L23.up_proj
 420  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 421  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L23.down_proj
 422  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L24.norms
 423  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 424  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L24.q_proj
 425  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L24.k_proj
 426  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L24.v_proj
 427  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 428  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 429  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 430  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 431  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=24 L24.K0
 432  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=24 L24.K0
 433  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L24.o_proj
 434  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 435  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L24.gate_proj
 436  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L24.up_proj
 437  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 438  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L24.down_proj
 439  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L25.norms
 440  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 441  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L25.q_proj
 442  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L25.k_proj
 443  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L25.v_proj
 444  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 445  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 446  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 447  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 448  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=25 L25.K0
 449  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=25 L25.K0
 450  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L25.o_proj
 451  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 452  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L25.gate_proj
 453  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L25.up_proj
 454  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 455  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L25.down_proj
 456  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L26.norms
 457  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 458  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L26.q_proj
 459  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L26.k_proj
 460  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L26.v_proj
 461  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 462  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 463  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 464  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 465  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=26 L26.K0
 466  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=26 L26.K0
 467  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L26.o_proj
 468  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 469  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L26.gate_proj
 470  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L26.up_proj
 471  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 472  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L26.down_proj
 473  VLOAD flags=00 dst=229376 src0=0 src1=0 n=4352 aux=0 L27.norms
 474  RMSN  flags=00 dst=16384 src0=0 src1=229376 n=2048 aux=1
 475  GEMV  flags=00 dst=32768 src0=16384 src1=0 n=2048 aux=16 L27.q_proj
 476  GEMV  flags=00 dst=49152 src0=16384 src1=0 n=1024 aux=16 L27.k_proj
 477  GEMV  flags=00 dst=57344 src0=16384 src1=0 n=1024 aux=16 L27.v_proj
 478  RMSN  flags=00 dst=32768 src0=32768 src1=233472 n=2048 aux=16
 479  RMSN  flags=00 dst=49152 src0=49152 src1=233600 n=1024 aux=8
 480  ROPE  flags=00 dst=32768 src0=32768 src1=0 n=2048 aux=16
 481  ROPE  flags=00 dst=49152 src0=49152 src1=0 n=1024 aux=8
 482  KVW   flags=00 dst=0 src0=49152 src1=57344 n=0 aux=27 L27.K0
 483  ATTN  flags=00 dst=65536 src0=32768 src1=0 n=2048 aux=27 L27.K0
 484  GEMV  flags=01 dst=0 src0=65536 src1=0 n=2048 aux=16 L27.o_proj
 485  RMSN  flags=00 dst=16384 src0=0 src1=231424 n=2048 aux=1
 486  GEMV  flags=00 dst=81920 src0=16384 src1=0 n=6144 aux=16 L27.gate_proj
 487  GEMV  flags=00 dst=131072 src0=16384 src1=0 n=6144 aux=16 L27.up_proj
 488  SILU  flags=00 dst=180224 src0=81920 src1=131072 n=6144 aux=0
 489  GEMV  flags=01 dst=0 src0=180224 src1=0 n=2048 aux=48 L27.down_proj
 490  VLOAD flags=00 dst=229376 src0=0 src1=0 n=2048 aux=0 final_norm
 491  RMSN  flags=20 dst=30720 src0=14336 src1=229376 n=2048 aux=1
 492  GEMV  flags=22 dst=0 src0=30720 src1=0 n=151936 aux=16 lm_head
 493  END   flags=00 dst=0 src0=0 src1=0 n=0 aux=0
