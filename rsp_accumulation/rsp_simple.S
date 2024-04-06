## Auto-generated file, transpiled with RSPL
#define MATRIX_SIZE 1
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
    RSPQ_DefineCommand MatCopyIn, 12
  RSPQ_EndOverlayHeader

  RSPQ_BeginSavedState
    STATE_MEM_START:
    .align 3
    A: .ds.b 16
    .align 3
    B: .ds.b 16
    .align 3
    RET: .ds.b 32
    STATE_MEM_END:
  RSPQ_EndSavedState

.text

MatCopyIn:
  ori $s4, $zero, %lo(A)
  lui $at, 0xFF
  ori $at, $at, 0xFFFF
  and $a0, $a0, $at
  addiu $t2, $zero, 12
  lui $at, 0xFF
  ori $at, $at, 0xFFFF
  and $a1, $a1, $at
  lui $at, 0xFF
  ori $at, $at, 0xFFFF
  ori $t0, $zero, 15
  or $s0, $zero, $a1
  jal DMAExec ## Args: $t0, $t1, $s0, $s4, $t2
  and $a2, $a2, $at
  ori $t0, $zero, 15
  ori $s4, $zero, %lo(B)
  or $s0, $zero, $a2
  jal DMAExec ## Args: $t0, $t1, $s0, $s4, $t2
  addiu $t2, $zero, 12
  ori $at, $zero, %lo(A)
  lqv $v28, 0, 0, $at
  ori $at, $zero, %lo(B)
  lqv $v27, 0, 0, $at
  ori $at, $zero, %lo(RET)
  ori $t0, $zero, 31
  addiu $t2, $zero, -32756
  vmudh $v26, $v28, $v27.v
  or $s0, $zero, $a0
  vsar $v26, COP2_ACC_MD
  vsar $v25, COP2_ACC_HI
  ori $s4, $zero, %lo(RET)
  sqv $v26, 0, 16, $at
  jal DMAExec ## Args: $t0, $t1, $s0, $s4, $t2
  sqv $v25, 0, 0, $at
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
