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

    HyrbidAgent(ALEInterface* ale, NervousSystem& neural_net);
    HyrbidAgent(ALEInterface* ale, NervousSystem& neural_net, Index update_rate);
    ~HyrbidAgent();

    void Configure(const float *parameters);
    void Reset();

  protected:
    Action Act();

  protected:
    NervousSystem& neural_net_;
    std::vector<std::uint8_t> full_screen_;
    std::vector<float> buffer_screen1_;
    std::vector<float> buffer_screen2_;
    std::vector<float> downsized_screen_;
    std::size_t update_rate_;
    bool is_configured_;
};

} // End alectrnn namespace

#endif /* ALECTRNN_HYBRID_AGENT_H_ */