/*
 * A derived class from PlayerAgent, it uses a hybrid nervous system
 * as the basis for Action decisions.
 */

#ifndef ALECTRNN_NERVOUS_SYSTEM_AGENT_H_
#define ALECTRNN_NERVOUS_SYSTEM_AGENT_H_

#include <cstddef>
#include <vector>
#include <ale_interface.hpp>
#include "player_agent.hpp"
#include "../common/nervous_system.hpp"
#include "../common/state_logger.hpp"
#include "../common/screen_logger.hpp"

namespace alectrnn {

class NervousSystemAgent : public PlayerAgent {
  public:
    typedef std::size_t Index;

    NervousSystemAgent(ALEInterface* ale, nervous_system::NervousSystem<float>& neural_net);
    NervousSystemAgent(ALEInterface* ale, nervous_system::NervousSystem<float>& neural_net, 
        Index update_rate, bool is_logging);
    ~NervousSystemAgent();

    void Configure(const float *parameters);
    void Reset();
    const nervous_system::StateLogger<float>& GetLog() const;
    const ScreenLogger<float>& GetScreenLog() const;

  protected:
    Action Act();

  protected:
    nervous_system::NervousSystem<float>& neural_net_;
    nervous_system::StateLogger<float> log_;
    std::vector<std::uint8_t> grey_screen_;
    std::vector<std::uint8_t> color_screen_;
    std::vector<float> buffer_screen1_;
    std::vector<float> buffer_screen2_;
    std::vector<float> downsized_screen_;
    std::size_t update_rate_;
    bool is_configured_;
    bool is_logging_;
    ScreenLogger<float> screen_log_;
};

} // End alectrnn namespace

#endif /* ALECTRNN_NERVOUS_SYSTEM_AGENT_H_ */