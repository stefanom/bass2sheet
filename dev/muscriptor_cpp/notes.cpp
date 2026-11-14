// Runs MuScriptor (Rouard et al., 2026) through muscriptor.cpp, its C++/ggml port, and writes the notes it finds
// in each recording as CSV: start,end,pitch,instrument. See CMakeLists.txt beside this file and
// dev/muscriptor_eval.py, which calls it.
//
//   muscriptor_notes MODEL.gguf any|NAME[,NAME...] IN.f32 OUT.csv [IN.f32 OUT.csv ...]
//
// IN.f32 is the whole recording as raw 32-bit floats, mono, 16 kHz. The second argument is `any` for whatever
// instruments the model hears, or the instrument groups it may write and no others (electric_bass,acoustic_bass).
// A job whose OUT.csv exists is skipped. With MUSCRIPTOR_AS_REFERENCE set, the port's two additions to the
// reference decoding (a lead-in of silence, a gate on silent stretches) are turned off.

#include "muscriptor/muscriptor.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

static std::vector<float> readFloats(const char* inPath)
{
    std::ifstream in(inPath, std::ios::binary | std::ios::ate);
    if (!in) {
        return {};
    }
    std::vector<float> samples(static_cast<std::size_t>(in.tellg()) / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(samples.data()), static_cast<std::streamsize>(samples.size() * sizeof(float)));
    return samples;
}

int main(int argc, char** argv)
{
    if (argc < 5 || argc % 2 == 0) {
        std::fprintf(stderr, "usage: %s MODEL.gguf any|NAME[,NAME...] IN.f32 OUT.csv [IN.f32 OUT.csv ...]\n", argv[0]);
        return 2;
    }
    msl::TranscribeOptions options;
    for (std::string_view rest = argv[2]; rest != "any" && !rest.empty();) {
        const std::string_view name = rest.substr(0, rest.find(','));
        rest.remove_prefix(name.size() == rest.size() ? rest.size() : name.size() + 1);
        bool known = false;
        for (const msl::InstrumentGroup group : msl::allInstrumentGroups()) {
            if (msl::instrumentName(group) == name) {
                options.instruments.push_back(group);
                known = true;
            }
        }
        if (!known) {
            std::fprintf(stderr, "no instrument group is called %.*s\n", static_cast<int>(name.size()), name.data());
            return 2;
        }
    }
    if (std::getenv("MUSCRIPTOR_AS_REFERENCE") != nullptr) {
        options.lead_in.enabled = false;
        options.silence_gate.enabled = false;
    }

    auto model = msl::Transcriber::load(argv[1]);
    if (!model) {
        std::fprintf(stderr, "cannot load %s: %s\n", argv[1], msl::describe(model.error()));
        return 1;
    }
    std::fprintf(stderr, "MuScriptor runs on %s (%s)\n", model->device().name.c_str(), model->device().backend.c_str());

    for (int k = 3; k + 1 < argc; k += 2) {
        const std::string out = argv[k + 1];
        if (std::filesystem::exists(out)) {
            continue;
        }
        const std::vector<float> samples = readFloats(argv[k]);
        if (samples.empty()) {
            std::fprintf(stderr, "cannot read %s\n", argv[k]);
            return 1;
        }
        const auto began = std::chrono::steady_clock::now();
        const auto notes = model->transcribe(samples, options);
        if (!notes) {
            std::fprintf(stderr, "%s: %s\n", argv[k], msl::describe(notes.error()));
            return 1;
        }
        const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        {
            std::ofstream csv(out + ".part");
            csv << "start,end,pitch,instrument\n";
            char line[128];
            for (const msl::Note& note : *notes) {
                const std::string label = note.is_drum ? "drums" : msl::instrumentLabel(note.program);
                std::snprintf(line, sizeof line, "%.4f,%.4f,%d,%s\n", note.onset, note.offset, note.pitch, label.c_str());
                csv << line;
            }
        }
        std::filesystem::rename(out + ".part", out);
        std::fprintf(stderr, "%s: %zu notes, %.0f s of sound in %.0f s\n", std::filesystem::path(out).filename().c_str(),
                     notes->size(), static_cast<double>(samples.size()) / msl::Transcriber::SAMPLE_RATE, took);
    }
    return 0;
}
