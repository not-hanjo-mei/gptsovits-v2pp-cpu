    .intel_syntax noprefix
    .text
    .globl conv16_asm
    .type conv16_asm, @function
# void conv16_asm(const float* X, int64_t L, const float* W, int64_t C,
#                 int64_t Ci, int64_t K, int64_t dil, float* acc)
#   rdi = X (row 0 already offset to src0), rsi = L (floats),
#   rdx = W (already offset to o0*C), rcx = C (floats),
#   r8 = Ci (even), r9 = K, [rsp+8] = dil, [rsp+16] = acc
# acc: 64 floats [o][16], pre-initialized (bias); updated in place.
# Computes acc[o][p] += sum_i sum_kk W[(o*C) + i*K + kk] * X[i*L + kk*dil + p]
# Register plan: ymm0-7 accumulators, ymm8-11 x vectors, ymm12/13
# broadcast temps. The i loop runs in pairs so each accumulator's
# two FMA hits per tap are separated by the other row's 8 FMAs.
conv16_asm:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r12, QWORD PTR [rsp + 64]      # acc (7 pushes + ret = 64)
    mov rax, QWORD PTR [rsp + 56]      # dil
    # load accumulators
    vmovups ymm0, YMMWORD PTR [r12]
    vmovups ymm1, YMMWORD PTR [r12 + 32]
    vmovups ymm2, YMMWORD PTR [r12 + 64]
    vmovups ymm3, YMMWORD PTR [r12 + 96]
    vmovups ymm4, YMMWORD PTR [r12 + 128]
    vmovups ymm5, YMMWORD PTR [r12 + 160]
    vmovups ymm6, YMMWORD PTR [r12 + 192]
    vmovups ymm7, YMMWORD PTR [r12 + 224]
    lea rsi, [rsi * 4]                 # L bytes
    lea rcx, [rcx * 4]                 # C bytes
    lea rbp, [rax * 4]                 # dil bytes (x tap stride)
    lea r13, [r9 * 4]                  # K bytes (weight row stride)
    mov r14, r8
    shr r14, 1                         # i pairs
    xor ebx, ebx                       # weight base offset for row i
    mov r10, rdi                       # row i ptr
    test r14, r14
    jz .Ldone
.Lpair:
    lea r11, [r10 + rsi]               # row j ptr
    xor eax, eax                       # x tap offset
    xor r15d, r15d                     # weight tap offset
.Lkk:
    vmovups ymm8, YMMWORD PTR [r10 + rax]
    vmovups ymm9, YMMWORD PTR [r10 + rax + 32]
    vmovups ymm10, YMMWORD PTR [r11 + rax]
    vmovups ymm11, YMMWORD PTR [r11 + rax + 32]
    lea rdi, [rdx + rbx]
    lea rdi, [rdi + r15]               # row i, o0 weight addr
    vbroadcastss ymm12, DWORD PTR [rdi]
    vfmadd231ps ymm0, ymm12, ymm8
    vfmadd231ps ymm1, ymm12, ymm9
    add rdi, rcx
    vbroadcastss ymm12, DWORD PTR [rdi]
    vfmadd231ps ymm2, ymm12, ymm8
    vfmadd231ps ymm3, ymm12, ymm9
    add rdi, rcx
    vbroadcastss ymm12, DWORD PTR [rdi]
    vfmadd231ps ymm4, ymm12, ymm8
    vfmadd231ps ymm5, ymm12, ymm9
    add rdi, rcx
    vbroadcastss ymm12, DWORD PTR [rdi]
    vfmadd231ps ymm6, ymm12, ymm8
    vfmadd231ps ymm7, ymm12, ymm9
    lea rdi, [rdx + rbx]
    lea rdi, [rdi + r15]
    add rdi, r13                       # row j weights (+ K bytes)
    vbroadcastss ymm12, DWORD PTR [rdi]
    vfmadd231ps ymm0, ymm12, ymm10
    vfmadd231ps ymm1, ymm12, ymm11
    add rdi, rcx
    vbroadcastss ymm12, DWORD PTR [rdi]
    vfmadd231ps ymm2, ymm12, ymm10
    vfmadd231ps ymm3, ymm12, ymm11
    add rdi, rcx
    vbroadcastss ymm12, DWORD PTR [rdi]
    vfmadd231ps ymm4, ymm12, ymm10
    vfmadd231ps ymm5, ymm12, ymm11
    add rdi, rcx
    vbroadcastss ymm12, DWORD PTR [rdi]
    vfmadd231ps ymm6, ymm12, ymm10
    vfmadd231ps ymm7, ymm12, ymm11
    add rax, rbp
    add r15, 4
    cmp r15, r13
    jb .Lkk
    lea r10, [r10 + rsi * 2]
    lea rbx, [rbx + r13 * 2]
    dec r14
    jnz .Lpair
.Ldone:
    vmovups YMMWORD PTR [r12], ymm0
    vmovups YMMWORD PTR [r12 + 32], ymm1
    vmovups YMMWORD PTR [r12 + 64], ymm2
    vmovups YMMWORD PTR [r12 + 96], ymm3
    vmovups YMMWORD PTR [r12 + 128], ymm4
    vmovups YMMWORD PTR [r12 + 160], ymm5
    vmovups YMMWORD PTR [r12 + 192], ymm6
    vmovups YMMWORD PTR [r12 + 224], ymm7
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
    .size conv16_asm, .-conv16_asm
    .section .note.GNU-stack,"",@progbits
