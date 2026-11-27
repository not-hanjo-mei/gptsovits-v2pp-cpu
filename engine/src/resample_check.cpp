// M3-3d stage (a) check: replicate Genie's ReferenceAudio
// preparation (load_audio at 32 kHz + 0.3 s trailing silence,
// then soxr 'hq' downsample to 16 kHz) with the host's own
// WAV reader and sinc resampler, and compare both arrays
// against the oracle .npy dumps of ReferenceAudio's fields.
//
// Usage: resample_check REF.wav REF32.npy REF16.npy
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "resample.h"
#include "wavread.h"

// Minimal .npy reader: v1/v2 header, '<f4' payload only.
static bool read_npy_f4(const std::string& path, std::vector<float>& out, std::vector<int64_t>& shape) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> buf;
    uint8_t chunk[65536];
    size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
    fclose(f);
    if (buf.size() < 10 || memcmp(buf.data(), "\x93NUMPY", 6) != 0) return false;
    size_t hdr_off, hdr_len;
    if (buf[6] == 1) {
        hdr_len = buf[8] | (buf[9] << 8);
        hdr_off = 10;
    } else {
        hdr_len = (size_t)buf[8] | ((size_t)buf[9] << 8) | ((size_t)buf[10] << 16) | ((size_t)buf[11] << 24);
        hdr_off = 12;
    }
    std::string hdr((const char*)buf.data() + hdr_off, hdr_len);
    if (hdr.find("'<f4'") == std::string::npos && hdr.find("\"<f4\"") == std::string::npos &&
        hdr.find("'<f4'") == std::string::npos && hdr.find("<f4") == std::string::npos)
        return false;
    auto sp = hdr.find("'shape':");
    if (sp == std::string::npos) return false;
    auto lp = hdr.find('(', sp), rp = hdr.find(')', lp);
    std::string dims = hdr.substr(lp + 1, rp - lp - 1);
    shape.clear();
    int64_t total = 1;
    for (size_t i = 0; i < dims.size();) {
        while (i < dims.size() && (dims[i] < '0' || dims[i] > '9')) i++;
        if (i >= dims.size()) break;
        int64_t v = 0;
        while (i < dims.size() && dims[i] >= '0' && dims[i] <= '9') v = v * 10 + (dims[i++] - '0');
        shape.push_back(v);
        total *= v;
    }
    size_t data_off = hdr_off + hdr_len;
    if (buf.size() < data_off + (size_t)total * 4) return false;
    out.resize((size_t)total);
    memcpy(out.data(), buf.data() + data_off, (size_t)total * 4);
    return true;
}

static void compare(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
    if (got.size() != ref.size()) {
        std::printf("%s: SIZE got %zu ref %zu\n", name, got.size(), ref.size());
        return;
    }
    double dot = 0, gg = 0, rr = 0, maxerr = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = (double)got[i] - (double)ref[i];
        dot += (double)got[i] * ref[i];
        gg += (double)got[i] * got[i];
        rr += (double)ref[i] * ref[i];
        if (std::fabs(d) > maxerr) maxerr = std::fabs(d);
    }
    double corr = dot / std::sqrt(std::max(1e-30, gg * rr));
    std::printf("%s: n %zu corr %.6f maxerr %.6g\n", name, got.size(), corr, maxerr);
}

static void write_npy_f4(const std::string& path, const std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                      std::to_string(v.size()) + ",), }";
    while ((10 + hdr.size() + 1) % 64) hdr.push_back(' ');
    hdr.push_back('\n');
    uint8_t magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (uint8_t)(hdr.size() & 0xff),
                         (uint8_t)(hdr.size() >> 8)};
    fwrite(magic, 1, 10, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    fwrite(v.data(), 4, v.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: resample_check REF.wav REF32.npy REF16.npy [OUT16.npy]\n");
        return 2;
    }
    refaudio::WavData w;
    std::string err;
    if (!refaudio::read_wav(argv[1], w, err)) {
        std::fprintf(stderr, "wav: %s\n", err.c_str());
        return 1;
    }
    std::printf("wav: sr %d frames %zu\n", w.sample_rate, w.mono.size());
    // Genie load_audio(path, 32000): resample if needed, then
    // append 0.3 s of silence at 32 kHz.
    std::vector<float> a32 = (w.sample_rate == 32000) ? w.mono
                                                      : refaudio::resample_sinc(w.mono, w.sample_rate, 32000);
    a32.insert(a32.end(), 9600, 0.0f);
    // ReferenceAudio: audio_16k = soxr(audio_32k_1d, 32k -> 16k).
    std::vector<float> a16 = refaudio::resample_sinc(a32, 32000, 16000);

    std::vector<float> ref32, ref16;
    std::vector<int64_t> sh;
    if (!read_npy_f4(argv[2], ref32, sh)) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    if (!read_npy_f4(argv[3], ref16, sh)) { std::fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
    compare("audio_32k", a32, ref32);
    compare("audio_16k", a16, ref16);
    if (argc == 5) write_npy_f4(argv[4], a16);
    return 0;
}
