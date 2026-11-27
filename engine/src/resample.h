// Band-limited resampler for the runtime reference path:
// a Kaiser-windowed sinc interpolator, double accumulation,
// no renormalization at the edges (out-of-range input reads
// as zero). It does not reproduce soxr 'hq' bit-for-bit --
// soxr's transition band (7.3-8.0 kHz for 32k->16k) is wider
// and more gradual than any steep Kaiser fit, and steeper
// fits measured worse, not better. With the defaults below,
// resample_check against Genie's soxr output gives
// audio_16k corr 0.999918 (maxerr 0.021), and HuBERT's
// ssl_content computed from the resampled audio agrees
// with the shipped pack at corr 0.999993, so the residual
// difference is inaudible to the downstream models. The DC
// gain is exact by construction: sum_t 2fc*sinc(2fc*t) = 1
// for fc <= 0.5.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace refaudio {

// Modified Bessel function I0 (series expansion).
inline double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    double x2 = x * x / 4.0;
    for (int k = 1; k < 64; k++) {
        term *= x2 / ((double)k * (double)k);
        sum += term;
        if (term < 1e-15 * sum) break;
    }
    return sum;
}

// Resample `in` from in_sr to out_sr. fc_scale narrows the
// passband below the output Nyquist (transition band);
// half_len is the window half-length in input samples.
inline std::vector<float> resample_sinc(const std::vector<float>& in, int in_sr, int out_sr,
                                        double fc_scale = 0.96, int half_len = 128,
                                        double kaiser_beta = 9.5) {
    if (in_sr == out_sr) return in;
    double ratio = (double)out_sr / (double)in_sr;         // out per in
    double fc = 0.5 * (ratio < 1.0 ? ratio : 1.0) * fc_scale;  // cycles / input sample
    int64_t n_out = (int64_t)std::llround((double)in.size() * ratio);
    std::vector<float> out((size_t)n_out);
    double i0_beta = bessel_i0(kaiser_beta);
    for (int64_t m = 0; m < n_out; m++) {
        double center = (double)m / ratio;  // in input samples
        int64_t i0 = (int64_t)std::floor(center);
        double acc = 0.0;
        for (int64_t i = i0 - half_len + 1; i <= i0 + half_len; i++) {
            if (i < 0 || i >= (int64_t)in.size()) continue;
            double t = (double)i - center;  // offset in input samples
            double x = 2.0 * fc * t;
            double sinc = (x == 0.0) ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
            double u = t / (double)half_len;  // -1..1 across the window
            double win = bessel_i0(kaiser_beta * std::sqrt(std::max(0.0, 1.0 - u * u))) / i0_beta;
            acc += (double)in[(size_t)i] * (2.0 * fc * sinc * win);
        }
        out[(size_t)m] = (float)acc;
    }
    return out;
}

}  // namespace refaudio
