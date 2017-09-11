#include <cstddef>
#include <cstdint>
#include "ctrnn_agent.h"
#include "player_agent.h"
#include "../common/network_generator.h"
#include "../common/screen_preprocessing.h"
#include <ale_interface.hpp>

namespace alectrnn {

CtrnnAgent::CtrnnAgent(ALEInterface* ale, double* weights,
      std::size_t num_neurons) : PlayerAgent(ale) {
}

CtrnnAgent::~CtrnnAgent() {
}

Action CtrnnAgent::Act() {

}

}

// use ALE to get # sensor inputs???... not sure what obj might give though...
// need to split up weights so that both chunks gets the right number of params
// params also need to be parsed into vectors for the other things (gain, bias, tau)
//    std::vector<std::vector<ctrnn::InEdge> > node_neighbors = All2AllNetwork(num_nodes, ??);
//    std::vector<std::vector<ctrnn::InEdge> > node_sensors = FullSensorNetwork(num_nodes, num_sensors, ??);


//    ale->getScreenGrayscale(full_gray_screen);
    //get screen size replace constants

//    ResizeGrayScreen(210, 160, 84, 84, full_gray_screen, small_gray_screen);
    //  pass video to Agent
