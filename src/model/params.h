#ifndef LANG_ID_PARAM_H
#define LANG_ID_PARAM_H

#include "insnet/insnet.h"

class ModelParams : public insnet::TunableParamCollection
#if USE_GPU
, public insnet::cuda::TransferableComponents
#endif
{
public:
    void init(const insnet::Vocab &vocab, int dim, int word_layer, int word_head, int seg_layer,
            int seg_head,
            int sent_layer,
            int max_len,
            int class_num) {
        emb.init(vocab, dim, true);
        word_enc.init(word_layer, dim, word_head, max_len);
        std::function<void(insnet::LSTMParams &, int)> init_lstm = [&] (insnet::LSTMParams &params,
                int layer) {
            params.init(dim, dim);
        };
        seg_enc.init(seg_layer, dim, seg_head, max_len);
        sent_enc.init(sent_layer, init_lstm);
        output.init(class_num, dim);
    }

    insnet::Embedding<insnet::SparseParam> emb;
    insnet::TransformerEncoderParams word_enc;
    insnet::TransformerEncoderParams seg_enc;
    insnet::ParamArray<insnet::LSTMParams> sent_enc;
    insnet::LinearParams output;

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(emb, word_enc, seg_enc, sent_enc, output);
    }

#if USE_GPU
    std::vector<insnet::cuda::Transferable *> transferablePtrs() override {
        return {&emb, &seg_enc, &sent_enc, &word_enc, &output};
    }
#endif

protected:
    virtual std::vector<insnet::TunableParam *> tunableComponents() override {
        return {&emb, &seg_enc, &sent_enc, &word_enc, &output};
    }
};

#endif
