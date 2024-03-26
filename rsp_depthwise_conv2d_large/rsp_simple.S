## Auto-generated file, transpiled with RSPL
#define IN_SIZE 90
#define W_SIZE 9
#define OSIZE 80
#include <rsp_queue.inc>
#include <rdpq_macros.h>

.set noreorder
.set noat
.set nomacro

#undef zero
#undef at
#undef v0
#undef v1
#undef a0
#undef a1
#undef a2
#undef a3
#undef t0
#undef t1
#undef t2
#undef t3
#undef t4
#undef t5
#undef t6
#undef t7
#undef s0
#undef s1
#undef s2
#undef s3
#undef s4
#undef s5
#undef s6
#undef s7
#undef t8
#undef t9
#undef k0
#undef k1
#undef gp
#undef sp
#undef fp
#undef ra
.equ hex.$zero, 0
.equ hex.$at, 1
.equ hex.$v0, 2
.equ hex.$v1, 3
.equ hex.$a0, 4
.equ hex.$a1, 5
.equ hex.$a2, 6
.equ hex.$a3, 7
.equ hex.$t0, 8
.equ hex.$t1, 9
.equ hex.$t2, 10
.equ hex.$t3, 11
.equ hex.$t4, 12
.equ hex.$t5, 13
.equ hex.$t6, 14
.equ hex.$t7, 15
.equ hex.$s0, 16
.equ hex.$s1, 17
.equ hex.$s2, 18
.equ hex.$s3, 19
.equ hex.$s4, 20
.equ hex.$s5, 21
.equ hex.$s6, 22
.equ hex.$s7, 23
.equ hex.$t8, 24
.equ hex.$t9, 25
.equ hex.$k0, 26
.equ hex.$k1, 27
.equ hex.$gp, 28
.equ hex.$sp, 29
.equ hex.$fp, 30
.equ hex.$ra, 31
#define vco 0
#define vcc 1
#define vce 2

.data
  RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand DMAInWeights, 4
    RSPQ_DefineCommand DMAInInputs, 4
    RSPQ_DefineCommand DepthConv, 4
  RSPQ_EndOverlayHeader

  RSPQ_BeginSavedState
    .align 3
    MAT_RES: .ds.b 1280
    .align 3
    MAT_IN: .ds.b 1440
    .align 3
    MAT_W: .ds.b 144
  RSPQ_EndSavedState

.text

DMAInWeights:
  lui $at, 0xFF
  or $t2, $zero, $zero
  ori $at, $at, 0xFFFF
  ori $s4, $zero, %lo(MAT_W)
  ori $t0, $zero, 143
  and $a0, $a0, $at
  jal DMAExec
  or $s0, $zero, $a0
  j RSPQ_Loop
  nop
DMAInInputs:
  lui $at, 0xFF
  ori $at, $at, 0xFFFF
  and $a0, $a0, $at
  ori $s4, $zero, %lo(MAT_IN)
  or $t2, $zero, $zero
  ori $t0, $zero, 1439
  jal DMAExec
  or $s0, $zero, $a0
  j RSPQ_Loop
  nop
DepthConv:
  lui $at, 0xFF
  ori $at, $at, 0xFFFF
  and $a0, $a0, $at
  ori $at, $zero, %lo(MAT_W)
  lqv $v28, 0, 0, $at
  lqv $v27, 0, 16, $at
  lqv $v26, 0, 32, $at
  lqv $v24, 0, 64, $at
  lqv $v23, 0, 80, $at
  lqv $v21, 0, 112, $at
  lqv $v25, 0, 48, $at
  ori $s7, $zero, %lo(MAT_IN)
  lqv $v20, 0, 128, $at
  lqv $v22, 0, 96, $at
  ori $s6, $zero, %lo(MAT_RES)
  or $s5, $zero, $zero
  jal DMAWaitIdle
  nop
  LABEL_0001:
  or $s4, $zero, $zero
  LABEL_0003:
  or $s3, $zero, $s7
  lqv $v19, 0, 0, $s3
  vmudn $v18, $v19, $v28.v
  lqv $v19, 0, 16, $s3
  vmadn $v18, $v19, $v27.v
  lqv $v19, 0, 32, $s3
  addiu $s3, $s3, 288
  vmadn $v18, $v19, $v26.v
  lqv $v19, 0, 0, $s3
  vmadn $v18, $v19, $v25.v
  lqv $v19, 0, 16, $s3
  vmadn $v18, $v19, $v24.v
  lqv $v19, 0, 32, $s3
  addiu $s3, $s3, 288
  vmadn $v18, $v19, $v23.v
  lqv $v19, 0, 0, $s3
  vmadn $v18, $v19, $v22.v
  lqv $v19, 0, 16, $s3
  vmadn $v18, $v19, $v21.v
  lqv $v19, 0, 32, $s3
  vmadn $v18, $v19, $v20.v
  sqv $v18, 0, 0, $s6
  addiu $s7, $s7, 16
  addiu $s4, $s4, 1
  sltiu $at, $s4, 16
  bne $at, $zero, LABEL_0003
  addiu $s6, $s6, 16
  LABEL_0004:
  addiu $s7, $s7, 32
  addiu $s5, $s5, 1
  sltiu $at, $s5, 5
  bne $at, $zero, LABEL_0001
  nop
  LABEL_0002:
  or $s0, $zero, $a0
  ori $t0, $zero, 1279
  ori $s4, $zero, %lo(MAT_RES)
  addiu $t2, $zero, -32756
  jal DMAExec
  nop
  j RSPQ_Loop
  nop

#define zero $0
#define v0 $2
#define v1 $3
#define a0 $4
#define a1 $5
#define a2 $6
#define a3 $7
#define t0 $8
#define t1 $9
#define t2 $10
#define t3 $11
#define t4 $12
#define t5 $13
#define t6 $14
#define t7 $15
#define s0 $16
#define s1 $17
#define s2 $18
#define s3 $19
#define s4 $20
#define s5 $21
#define s6 $22
#define s7 $23
#define t8 $24
#define t9 $25
#define k0 $26
#define k1 $27
#define gp $28
#define sp $29
#define fp $30
#define ra $31

.set at
.set macro
