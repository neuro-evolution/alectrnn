#ifndef ALECTRNN_CTRNN_AGENT_H_
#define ALECTRNN_CTRNN_AGENT_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <ale_interface.hpp>
#include "player_agent.h"
#include "../common/network_generator.h"

namespace alectrnn {

class CtrnnAgent : public PlayerAgent {
  public:
    CtrnnAgent(ALEInterface* ale, double* weights, std::size_t);
    ~CtrnnAgent();

  protected:
    Action Act();

  private:
    std::vector<std::vector<ctrnn::InEdge> > node_sensors_;
    std::vector<std::vector<ctrnn::InEdge> > node_neighbors_;
    std::vector<std::uint8_t> full_gray_screen_;
    std::vector<std::uint8_t> downsized_gray_screen_;
};

}

#endif /* ALECTRNN_CTRNN_AGENT_H_ */
