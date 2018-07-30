//
// Created by nathaniel on 5/13/18.
//

#ifndef ALECTRNN_SOFT_MAX_AGENT_HPP
#define ALECTRNN_SOFT_MAX_AGENT_HPP

#include <ale_interface.hpp>
#include <random>
#include "nervous_system_agent.hpp"
#include "../nervous_system/nervous_system.hpp"

namespace alectrnn {

class SoftMaxAgent : public NervousSystemAgent {
  public:
    typedef NervousSystemAgent super_type;
    typedef typename super_type::Index Index;

    SoftMaxAgent(ALEInterface* ale,
                 nervous_system::NervousSystem<float>& neural_net,
                 Index update_rate, bool is_logging, int seed)
        : super_type(ale, neural_net, update_rate, is_logging),
          rng_(seed), rand_real_(0.0, 1.0) {
    }

    void seed(int new_seed);

  protected:
    Action Act();
    Action GetActionFromNervousSystem();

  protected:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> rand_real_;
};

} // end alectrnn namespace

#endif //ALECTRNN_SOFT_MAX_AGENT_HPP
