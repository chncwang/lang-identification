#ifndef INSNET_BENCHMARK_DATA_MANAGER_H
#define INSNET_BENCHMARK_DATA_MANAGER_H

#include <atomic>
#include <thread>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <codecvt>
#include <fstream>
#include <iterator>
#include <regex>
#include <iostream>
#include <utility>
#include <atomic>
#include <mutex>
#include "conversation_structure.h"
#include "tinyutf8.h"
#include "def.h"
#include "fmt/core.h"
#include "insnet/insnet.h"

enum ParsingState {
    IN_WORD = 0,
    IN_SPACE = 1,
};

inline bool between(char32_t ch, char32_t begin, char32_t end) {
    if (begin > end) {
        std::cerr << begin << end << std::endl;
        abort();
    }
    return ch >= begin && ch <= end;
}

inline bool isCJK(char32_t ch) {
    return between(ch, 0x4e00, 0x9fef) || between(ch, 0x3400, 0x4dbf) ||
        between(ch, 0x20000, 0x2a6df) || between(ch, 0x2a700, 0x2b73f) ||
        between(ch, 0x2b740, 0x2b81f) || between(ch, 0x2b820, 0x2ceaf) ||
        between(ch, 0x2ceb0, 0x2ebef) || between(ch, 0x3007, 0x30ff) || between(ch, 0xf900, 0xfa6a)
        || between(ch, 0xff5f, 0xff9f);
}

inline std::string langName(const std::string &path) {
    std::string name = std::filesystem::path(path).filename();
    int dot_count = 0;
    int end = 0;
    for (int i = 0; i < name.size(); ++i) {
        if (name.at(i) == '.') {
            end = i;
            if (++dot_count == 3)
                break;
        }
    }
    return name.substr(0, end);
}

inline int wordId(std::unordered_map<std::string, int> &vocab, const std::string &str) {
    auto it = vocab.find(str);
    if (it == vocab.end()) {
        return vocab.at(UNK);
    } else {
        return it->second;
    }
}

inline static int max_word_len;

inline std::vector<int> splitIntoWords(const utf8_string &line,
        std::unordered_map<std::string, int> &vocab) {
    std::vector<int> ret;
    ParsingState state = ParsingState::IN_SPACE;
    int begin_i = 0;
    for (int i = 0; i < line.length(); ++i) {
        char32_t ch = line.at(i);
        if (ch == ' ') {
            if (state == ParsingState::IN_WORD) {
                state = ParsingState::IN_SPACE;
            }
        } else {
            if (state == ParsingState::IN_SPACE) {
                begin_i = i;
                state = ParsingState::IN_WORD;
                ret.push_back(vocab.at(WORD_SYMBOL));
            }
            auto str = line.substr(i, 1).cpp_str();
            int id = wordId(vocab, str);
            ret.push_back(id);
        }
    }
    return ret;
}

inline std::vector<int> filterSpaces(const utf8_string &line,
        std::unordered_map<std::string, int> &vocab) {
    std::vector<int> ret;
    for (int i = 0; i < line.length(); ++i) {
        char32_t ch = line.at(i);
        if (ch != ' ') {
            auto str = line.substr(i, 1).cpp_str();
            ret.push_back(wordId(vocab, str));
        }
    }
    return ret;
}

inline std::pair<std::vector<std::vector<int>>, std::vector<int>> readDataset(
        const std::string &dir_name,
        std::unordered_map<std::string, int> &vocab,
        std::unordered_map<std::string, int> &class_vocab,
        float rate = 1) {
    std::vector<std::vector<int>> sent_ret;
    std::vector<int> class_ret;
    int sent_num = 0;
    float warning_num = 0;

    for (const auto &entry : std::filesystem::directory_iterator(dir_name)) {
        std::cout << fmt::format("sent_num:{} rate:{}", sent_num, sent_num / 15462425.0f) << std::endl;
        std::string path = entry.path();
        std::ifstream ifs(path);
        std::string raw_line;
        std::string lang_name = langName(path);

        float style = 0;
        int local_sent_num = 0;
        int read_local_sent_num = 0;
        int read_sent_num = 0;
        while (std::getline(ifs, raw_line)) {
            ++sent_num;
            ++local_sent_num;
            if (local_sent_num % 100 >= rate * 100) {
                continue;
            }
            ++read_sent_num;
            ++read_local_sent_num;

            ParsingState state = ParsingState::IN_SPACE;
            utf8_string line(raw_line);
            int word_num = 0;
            int char_num = 0;
            int cjk_num = 0;
            for (int i = 0; i < line.length(); ++i) {
                char32_t ch = line.at(i);
                if (ch == ' ') {
                    if (state == ParsingState::IN_WORD) {
                        state = ParsingState::IN_SPACE;
                    }
                } else {
                    ++char_num;
                    if (state == ParsingState::IN_SPACE) {
                        state = ParsingState::IN_WORD;
                        ++word_num;
                    }
                }

                if (isCJK(ch)) {
                    cjk_num++;
                    if (cjk_num > 0.25 * line.length()) {
                        break;
                    }
                }
            }

            if (word_num * 20 > char_num && cjk_num < char_num * 0.25) {
                if (style / read_local_sent_num > 1.5) {
                    warning_num += 1;
                    std::cout << fmt::format("filename:{} line:{} word_num:{} char_num:{} warning_rate:{} style:{} cjk_num:{}",
                            path, raw_line, word_num, char_num, warning_num / read_sent_num,
                            style/ read_local_sent_num, cjk_num) << std::endl;
                }
                style += 1;
                sent_ret.push_back(splitIntoWords(line, vocab));
            } else {
                if (style / read_local_sent_num < 1.5) {
                    warning_num += 1;
                    std::cout << fmt::format("filename:{} line:{} word_num:{} char_num:{} warning_rate:{} style:{} cjk_num:{}",
                            path, raw_line, word_num, char_num, warning_num / read_sent_num,
                            style/ read_local_sent_num, cjk_num) << std::endl;
                }
                style += 2;
                sent_ret.push_back(filterSpaces(line, vocab));
            }
            int class_id = class_vocab.at(lang_name);
            class_ret.push_back(class_id);
        }
        if (local_sent_num > 0) {
            std::cout << fmt::format("filename:{} total style:{}", path, style/read_local_sent_num) <<
                std::endl;
        }
    }

    return std::make_pair(sent_ret, class_ret);
}

inline std::vector<std::string> charList(const std::string &dir, int cutoff = 0, float rate = 1) {
    std::vector<std::string> ret;
    std::unordered_map<std::string, int> word_stat;
    int sent_num = 0;

    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        std::string path = entry.path();
        std::ifstream ifs(path);
        std::string raw_line;

        while (std::getline(ifs, raw_line)) {
            ++sent_num;
            if (sent_num % 100 >= rate * 100) {
                continue;
            }
            utf8_string line(raw_line);
            for (int i = 0; i < line.length(); ++i) {
                std::string c = line.substr(i, 1).cpp_str();
                auto it = word_stat.find(c);
                if (it == word_stat.end()) {
                    word_stat.insert(std::make_pair(c, 1));
                } else {
                    word_stat.at(c) += 1;
                }
            }
        }
        std::cout << fmt::format("rate:{} size:{}", (float)sent_num / 15462425, word_stat.size())
            << std::endl;
    }

    for (const auto &it : word_stat) {
        if (it.second > cutoff) {
            ret.push_back(it.first);
        }
    }
    ret.push_back(UNK);
    ret.push_back(WORD_SYMBOL);

    return ret;
}

inline std::vector<std::string> classList(const std::string &dir) {
    std::vector<std::string> ret;

    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        std::string path = entry.path();
        std::ifstream ifs(path);
        std::string raw_line;

        auto lang_name = langName(path);
        std::cout << "lang:" << lang_name << std::endl;
        ret.push_back(lang_name);
    }

    return ret;
}
#endif
