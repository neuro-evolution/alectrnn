/*
 * A derived class from PlayerAgent, it uses a hybrid nervous system
 * as the basis for Action decisions.
 */

#ifndef ALECTRNN_HYBRID_AGENT_H_
#define ALECTRNN_HYBRID_AGENT_H_

#include <cstddef>
#include <vector>
#include <ale_interface.hpp>
#include "player_agent.hpp"
#include "../common/nervous_system.hpp"

namespace alectrnn {

class HyrbidAgent : public PlayerAgent {
  public:
    typedef std::size_t Index;

    HyrbidAgent(ALEInterface* ale, NervousSystem* neural_net);
    ~HyrbidAgent();

    void Configure(const float *parameters);
    void Reset();

  protected:
    Action Act();

  protected:
    NervousSystem *neural_net_;
    std::vector<float> buffer_screen_;
    std::vector<float> full_screen_;
    std::vector<float> downsized_screen_;
    std::size_t input_screen_width_; // can get scrn height/width from neural net input shape
    std::size_t input_screen_height_; // can get scrn height/width from neural net input shape
    std::size_t update_rate_;
    bool use_color_;
    bool is_configured_;
};

} // End alectrnn namespace

#endif /* ALECTRNN_HYBRID_AGENT_H_ */