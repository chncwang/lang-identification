#include "cxxopts.hpp"
#include "insnet/insnet.h"
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iomanip>
#include <array>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include <queue>
#include "conversation_structure.h"
#include "data_manager.h"
#include "def.h"
#include "common.h"
#include "model/params.h"
#include "model/model.h"
#include <iomanip>

using cxxopts::Options;
using std::string;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::ostringstream;
using std::priority_queue;
using std::unordered_map;
using std::unordered_set;
using std::unique_ptr;
using std::make_unique;
using std::move;
using std::pair;
using std::make_pair;
using std::default_random_engine;
using std::shuffle;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::duration;
using insnet::dtype;
using insnet::Vocab;
using insnet::Graph;
using insnet::Node;
using insnet::Profiler;

int main(int argc, const char *argv[]) {
    ModelParams params;
    Vocab vocab, class_vocab;

    Options options("lang_id");
    options.add_options()
        ("model", "load model", cxxopts::value<string>()->default_value("./model"))
        ("corpus", "corpus dir", cxxopts::value<string>())
        ("batch_size", "batch size", cxxopts::value<int>()->default_value("1"))
        ("ratio", "ratio", cxxopts::value<float>()->default_value("1"));

    auto args = options.parse(argc, argv);
    loadModel(params, vocab, class_vocab, args["model"].as<string>());

    string dir = args["corpus"].as<string>();
    int batch_size = args["batch_size"].as<int>();
    float ratio = args["ratio"].as<float>();

    auto ret = evaluate(params, 0.1, dir, vocab, class_vocab, 64, batch_size, ratio);
    cout << "f1:" << ret.first << " acc:" << ret.second << endl;

    return 0;
}
