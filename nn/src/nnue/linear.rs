#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const LOG2_WEIGHT_SCALE: i32 = 6; // 2^6 = 64

/// Quantized linear layer with 8-bit weights and 32-bits bias
/// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#linear-layer-4
pub unsafe fn linear(
    num_inputs: usize,
    num_outputs: usize,
    input: *const i8,
    weight: *const i8,
    bias: *const i32,
    output: *mut i32,
) {
    if num_outputs == 1 {
        // special case when there is only one output (probably final layer)
        let mut outval = *bias;
        for i in 0..num_inputs {
            let val1 = *input.add(i);
            let val2 = *weight.add(i);
            outval += (val1 as i32) * (val2 as i32); // NOTE: it is important to do calculations in i32
        }
        // account for weight scaling
        *output = outval >> 4; // /16
        return;
    }

    const REGISTER_WIDTH: usize = 256 / 8;

    assert!(num_inputs % REGISTER_WIDTH == 0); // processing 32 elements at a time
    assert!(num_outputs % 4 == 0); // processing 4 elements at a time

    let num_in_chunks: usize = num_inputs / REGISTER_WIDTH;
    let num_out_chunks: usize = num_outputs / 4;

    for i in 0..num_out_chunks {
        let offset0 = (i * 4 + 0) * num_inputs;
        let offset1 = (i * 4 + 1) * num_inputs;
        let offset2 = (i * 4 + 2) * num_inputs;
        let offset3 = (i * 4 + 3) * num_inputs;

        let mut sum0 = _mm256_setzero_si256();
        let mut sum1 = _mm256_setzero_si256();
        let mut sum2 = _mm256_setzero_si256();
        let mut sum3 = _mm256_setzero_si256();

        for j in 0..num_in_chunks {
            let inp = _mm256_load_si256(input.add(j * REGISTER_WIDTH) as *const __m256i);

            let w0 = _mm256_load_si256(weight.add(offset0 + j * REGISTER_WIDTH) as *const __m256i);
            let w1 = _mm256_load_si256(weight.add(offset1 + j * REGISTER_WIDTH) as *const __m256i);
            let w2 = _mm256_load_si256(weight.add(offset2 + j * REGISTER_WIDTH) as *const __m256i);
            let w3 = _mm256_load_si256(weight.add(offset3 + j * REGISTER_WIDTH) as *const __m256i);

            m256_add_dpbusd_epi32(&mut sum0, inp, w0);
            m256_add_dpbusd_epi32(&mut sum1, inp, w1);
            m256_add_dpbusd_epi32(&mut sum2, inp, w2);
            m256_add_dpbusd_epi32(&mut sum3, inp, w3);
        }

        let bias = _mm_load_si128(bias.add(i * 4) as *const __m128i);

        // -------------------------------------
        // m256_haddx4
        // https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#m256_haddx4
        sum0 = _mm256_hadd_epi32(sum0, sum1);
        sum2 = _mm256_hadd_epi32(sum2, sum3);
        sum0 = _mm256_hadd_epi32(sum0, sum2);

        let sum128lo = _mm256_castsi256_si128(sum0);
        let sum128hi = _mm256_extracti128_si256(sum0, 1);

        let mut outval = _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
        // -------------------------------------

        // account for weight scaling
        outval = _mm_srai_epi32(outval, LOG2_WEIGHT_SCALE);

        _mm_store_si128(output.add(i * 4) as *mut __m128i, outval);
    }
}

/// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#m256_add_dpbusd_epi32
#[inline]
unsafe fn m256_add_dpbusd_epi32(acc: &mut __m256i, a: __m256i, b: __m256i) {
    #[cfg(target_feature = "avx512vnni,avx512vl")]
    {
        *acc = _mm256_dpbusd_epi32(*acc, a, b);
    }

    #[cfg(not(target_feature = "avx512vnni,avx512vl"))]
    {
        let mut product0 = _mm256_maddubs_epi16(a, b);
        let one = _mm256_set1_epi16(1);
        product0 = _mm256_madd_epi16(product0, one);
        *acc = _mm256_add_epi32(*acc, product0);
    }
}

/// Refresh accumulator
/// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#feature-transformer-2
pub unsafe fn linear_partial_refresh(
    num_inputs: usize,
    num_outputs: usize,
    active_rows: &[u16],
    weight: *const i16,
    bias: *const i16,
    output: *mut i16,
) {
    const REGISTER_WIDTH: usize = 256 / 16;
    const NUM_CHUNKS: usize = 16;

    assert!(num_inputs % REGISTER_WIDTH == 0); // processing 16 elements at a time
    assert!(NUM_CHUNKS == num_outputs / REGISTER_WIDTH); // we expect only 16 chunks

    let mut regs: [__m256i; NUM_CHUNKS] = unsafe { std::mem::zeroed() };

    // init registers with bias
    for i in 0..NUM_CHUNKS {
        regs[i] = _mm256_load_si256(bias.add(i * REGISTER_WIDTH) as *const __m256i);
    }

    // accumulate enabled rows
    for &a in active_rows {
        for i in 0..NUM_CHUNKS {
            regs[i] = _mm256_add_epi16(
                regs[i],
                _mm256_load_si256(
                    weight.add((a as usize) * num_outputs + i * REGISTER_WIDTH) as *const __m256i
                ),
            );
        }
    }

    // copy to output
    for i in 0..NUM_CHUNKS {
        _mm256_store_si256(output.add(i * REGISTER_WIDTH) as *mut __m256i, regs[i]);
    }
}

/// Update accumulator
/// https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#feature-transformer-2
pub unsafe fn linear_partial_update(
    num_inputs: usize,
    num_outputs: usize,
    added_rows: &[u16],
    removed_rows: &[u16],
    weight: *const i16,
    inout: *const i16,
) {
    const REGISTER_WIDTH: usize = 256 / 16;
    const NUM_CHUNKS: usize = 16;

    assert!(num_inputs % REGISTER_WIDTH == 0); // processing 16 elements at a time
    assert!(NUM_CHUNKS == num_outputs / REGISTER_WIDTH); // we expect only 16 chunks

    let mut regs: [__m256i; NUM_CHUNKS] = unsafe { std::mem::zeroed() };

    // copy all existing values into the registers
    for i in 0..NUM_CHUNKS {
        regs[i] = _mm256_load_si256(inout.add(i * REGISTER_WIDTH) as *const __m256i);
    }

    // subtract removed rows
    for &r in removed_rows {
        for i in 0..NUM_CHUNKS {
            regs[i] = _mm256_sub_epi16(
                regs[i],
                _mm256_load_si256(
                    weight.add((r as usize) * num_outputs + i * REGISTER_WIDTH) as *const __m256i
                ),
            );
        }
    }

    // add added rows
    for &a in added_rows {
        for i in 0..NUM_CHUNKS {
            regs[i] = _mm256_add_epi16(
                regs[i],
                _mm256_load_si256(
                    weight.add((a as usize) * num_outputs + i * REGISTER_WIDTH) as *const __m256i
                ),
            );
        }
    }

    // copy the result back
    for i in 0..NUM_CHUNKS {
        _mm256_store_si256(inout.add(i * REGISTER_WIDTH) as *mut __m256i, regs[i]);
    }
}
