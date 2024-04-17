## Auto-generated file, transpiled with RSPL
#define SCRATCH_SIZE 213
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
    RSPQ_DefineCommand DMAInWeights, 8
    RSPQ_DefineCommand DMAInInputs, 8
    RSPQ_DefineCommand DepthConv, 8
    RSPQ_DefineCommand SetArgs, 28
  RSPQ_EndOverlayHeader

  RSPQ_BeginSavedState
    .align 3
    DATA: .ds.b 3408
    .align 2
    addrX: .ds.b 4
    .align 2
    addrOut: .ds.b 4
    .align 2
    out_part_h_s: .ds.b 4
    .align 2
    out_w_s: .ds.b 4
    .align 2
    w_stride_slice_s: .ds.b 4
    .align 2
    w_slide_byte_offset_s: .ds.b 4
    .align 2
    h_slide_byte_offset_s: .ds.b 4
  RSPQ_EndSavedState

.text

DMAInWeights:
  or $t0, $zero, $a1
  lui $at, 0xFF
  ori $s4, $zero, %lo(DATA)
  ori $at, $at, 0xFFFF
  or $t2, $zero, $zero
  addiu $t0, $t0, -1
  and $a0, $a0, $at
  jal DMAExec
  or $s0, $zero, $a0
  j RSPQ_Loop
  nop
SetArgs:
  lui $at, 0xFF
  lw $s0, %lo(RSPQ_DMEM_BUFFER) - 12($gp)
  lw $s1, %lo(RSPQ_DMEM_BUFFER) - 8($gp)
  ori $s7, $zero, %lo(DATA)
  lw $s2, %lo(RSPQ_DMEM_BUFFER) - 4($gp)
  sw $a3, %lo(w_slide_byte_offset_s)($zero)
  ori $at, $at, 0xFFFF
  addu $s7, $s7, $s1
  sw $s7, %lo(addrX)($zero)
  addu $s7, $s7, $s2
  and $a0, $a0, $at
  sw $s0, %lo(h_slide_byte_offset_s)($zero)
  sw $a2, %lo(w_stride_slice_s)($zero)
  sw $s7, %lo(addrOut)($zero)
  sw $a1, %lo(out_w_s)($zero)
  j RSPQ_Loop
  sw $a0, %lo(out_part_h_s)($zero)
DMAInInputs:
  lui $at, 0xFF
  or $t0, $zero, $a1
  addiu $t0, $t0, -1
  lw $s4, %lo(addrX + 0)
  ori $at, $at, 0xFFFF
  or $t2, $zero, $zero
  and $a0, $a0, $at
  jal DMAExec
  or $s0, $zero, $a0
  j RSPQ_Loop
  nop
DepthConv:
  lui $at, 0xFF
  lw $s3, %lo(w_stride_slice_s + 0)
  ori $at, $at, 0xFFFF
  lw $s4, %lo(addrOut + 0)
  and $a0, $a0, $at
  ori $at, $zero, %lo(DATA)
  lw $s2, %lo(w_slide_byte_offset_s + 0)
  lqv $v20, 0, 128, $at
  or $fp, $zero, $s4
  lqv $v27, 0, 16, $at
  lw $s5, %lo(out_part_h_s + 0)
  or $t0, $zero, $a1
  lw $s7, %lo(addrX + 0)
  lw $s1, %lo(h_slide_byte_offset_s + 0)
  lqv $v22, 0, 96, $at
  lqv $v21, 0, 112, $at
  lqv $v25, 0, 48, $at
  lqv $v24, 0, 64, $at
  lqv $v23, 0, 80, $at
  or $s0, $zero, $s7
  or $sp, $zero, $zero
  lqv $v26, 0, 32, $at
  lqv $v28, 0, 0, $at
  jal DMAWaitIdle
  lw $s6, %lo(out_w_s + 0)
  LABEL_0001:
  or $k1, $zero, $zero
  LABEL_0003:
  or $k0, $zero, $s0
  lqv $v19, 0, 0, $k0
  vmudn $v18, $v19, $v28.v
  lqv $v19, 0, 16, $k0
  vmadn $v18, $v19, $v27.v
  lqv $v19, 0, 32, $k0
  addu $k0, $k0, $s2
  addu $s0, $s0, $s3
  vmadn $v18, $v19, $v26.v
  lqv $v19, 0, 0, $k0
  vmadn $v18, $v19, $v25.v
  lqv $v19, 0, 16, $k0
  vmadn $v18, $v19, $v24.v
  lqv $v19, 0, 32, $k0
  addu $k0, $k0, $s2
  addiu $k1, $k1, 1
  sltu $at, $k1, $s6
  vmadn $v18, $v19, $v23.v
  lqv $v19, 0, 0, $k0
  vmadn $v18, $v19, $v22.v
  lqv $v19, 0, 16, $k0
  vmadn $v18, $v19, $v21.v
  lqv $v19, 0, 32, $k0
  vmadn $v18, $v19, $v20.v
  sqv $v18, 0, 0, $fp
  bne $at, $zero, LABEL_0003
  addiu $fp, $fp, 16
  LABEL_0004:
  addiu $sp, $sp, 1
  sltu $at, $sp, $s5
  bne $at, $zero, LABEL_0001
  addu $s0, $s0, $s1
  LABEL_0002:
  addiu $t0, $t0, -1
  or $s0, $zero, $a0
  jal DMAExec
  addiu $t2, $zero, -32756
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