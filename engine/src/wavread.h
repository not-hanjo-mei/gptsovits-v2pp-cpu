// Minimal RIFF/WAV reader for reference audio: PCM 16/24/32
// integer and IEEE float, any channel count (channels are
// averaged to mono, like Genie's load_audio). Integer samples
// are scaled by 2^(bits-1), the libsndfile convention.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace refaudio {

struct WavData {
    int sample_rate = 0;
    std::vector<float> mono;  // channel-averaged, full scale ~[-1, 1]
};

inline bool read_wav(const std::string& path, WavData& out, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> buf;
    {
        uint8_t chunk[65536];
        size_t n;
        while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) buf.insert(buf.end(), chunk, chunk + n);
    }
    fclose(f);
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, buf.data() + o, 4); return v; };
    auto u16 = [&](size_t o) { uint16_t v; memcpy(&v, buf.data() + o, 2); return v; };
    if (buf.size() < 44 || memcmp(buf.data(), "RIFF", 4) != 0 || memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file";
        return false;
    }
    uint16_t fmt_tag = 0, channels = 0, bits = 0;
    uint32_t sr = 0;
    size_t data_off = 0, data_len = 0;
    for (size_t o = 12; o + 8 <= buf.size();) {
        uint32_t len = u32(o + 4);
        if (memcmp(buf.data() + o, "fmt ", 4) == 0) {
            fmt_tag = u16(o + 8);
            channels = u16(o + 10);
            sr = u32(o + 12);
            bits = u16(o + 22);
            if (fmt_tag == 0xFFFE && len >= 40)  // extensible: real tag in the SubFormat GUID
                fmt_tag = u16(o + 32);
        } else if (memcmp(buf.data() + o, "data", 4) == 0) {
            data_off = o + 8;
            data_len = len;
            break;
        }
        o += 8 + len + (len & 1);
    }
    if (!data_off || !channels || !sr) { err = "missing fmt/data chunk"; return false; }
    if (data_off + data_len > buf.size()) data_len = buf.size() - data_off;
    int bytes = (bits + 7) / 8;
    if (bytes < 1 || bytes > 8) { err = "unsupported bit depth"; return false; }
    size_t frames = data_len / ((size_t)bytes * channels);
    out.sample_rate = (int)sr;
    out.mono.assign(frames, 0.0f);
    const uint8_t* p = buf.data() + data_off;
    double scale = bits > 0 ? 1.0 / (double)(1ull << (bits - 1)) : 1.0;
    for (size_t i = 0; i < frames; i++) {
        double acc = 0.0;
        for (uint16_t c = 0; c < channels; c++) {
            const uint8_t* s = p + (i * channels + c) * (size_t)bytes;
            double v = 0.0;
            if (fmt_tag == 3) {  // IEEE float
                if (bytes == 4) { float fv; memcpy(&fv, s, 4); v = fv; }
                else { double dv; memcpy(&dv, s, 8); v = dv; }
            } else if (fmt_tag == 1) {  // PCM integer, little-endian, sign-extended
                int64_t iv = 0;
                for (int b = bytes - 1; b >= 0; b--) iv = (iv << 8) | s[b];
                int shift = 64 - bytes * 8;
                iv = (iv << shift) >> shift;
                v = (double)iv * scale;
            } else {
                err = "unsupported format tag";
                return false;
            }
            acc += v;
        }
        out.mono[i] = (float)(acc / channels);
    }
    return true;
}

}  // namespace refaudio
