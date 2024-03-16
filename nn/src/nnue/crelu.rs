#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Clipped ReLU activation function from i16 elements
/// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#int16---int8
pub unsafe fn crelu_16(size: usize, input: *const i16, output: *mut i8) {
    const IN_REG_WIDTH: usize = 256 / 16;
    const OUT_REG_WIDTH: usize = 256 / 8;

    debug_assert!(size % OUT_REG_WIDTH == 0); // processing 32 elements at a time

    let num_out_chunks = size / OUT_REG_WIDTH;

    let zero = _mm256_setzero_si256();
    const CONTROL: i32 = 0b11011000;

    for i in 0..num_out_chunks {
        let in0 = _mm256_load_si256(input.add((i * 2 + 0) * IN_REG_WIDTH) as *const __m256i);
        let in1 = _mm256_load_si256(input.add((i * 2 + 1) * IN_REG_WIDTH) as *const __m256i);

        let result =
            _mm256_permute4x64_epi64(_mm256_max_epi8(_mm256_packs_epi16(in0, in1), zero), CONTROL);

        _mm256_store_si256(output.add(i * OUT_REG_WIDTH) as *mut __m256i, result);
    }
}

/// Clipped ReLU activation function from i32 elements
/// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#int32---int8
pub unsafe fn crelu_32(size: usize, input: *const i32, output: *mut i8) {
    const IN_REG_WIDTH: usize = 256 / 32;
    const OUT_REG_WIDTH: usize = 256 / 8;

    debug_assert!(size % OUT_REG_WIDTH == 0); // processing 32 elements at a time

    let num_out_chunks = size / OUT_REG_WIDTH;

    let zero = _mm256_setzero_si256();
    let control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    for i in 0..num_out_chunks {
        let in0 = _mm256_packs_epi32(
            _mm256_load_si256(input.add((i * 4 + 0) * IN_REG_WIDTH) as *const __m256i),
            _mm256_load_si256(input.add((i * 4 + 1) * IN_REG_WIDTH) as *const __m256i),
        );
        let in1 = _mm256_packs_epi32(
            _mm256_load_si256(input.add((i * 4 + 2) * IN_REG_WIDTH) as *const __m256i),
            _mm256_load_si256(input.add((i * 4 + 3) * IN_REG_WIDTH) as *const __m256i),
        );

        let result = _mm256_permutevar8x32_epi32(
            _mm256_max_epi8(_mm256_packs_epi16(in0, in1), zero),
            control,
        );

        _mm256_store_si256(output.add(i * OUT_REG_WIDTH) as *mut __m256i, result);
    }
}
