/*
 * ale_generator.cpp
 *
 *  Created on: Sep 1, 2017
 *      Author: Nathaniel Rodriguez
 *
 * ALE or Arcade-Learning-Environment wrapper for Python.
 * It generates a new environment and returns the pointer to the environment
 * in a Python Capsule for use in almost every other class. ALE is passed to
 * both the agent and controller classes for access to the ALE environment.
 *
 */

#include <Python.h>
#include <ale_interface.hpp>
#include <iostream>
#include "ale_generator.hpp"

static void DeleteALE(PyObject *ale_capsule) {
  delete (ALEInterface *)PyCapsule_GetPointer(ale_capsule, "ale_generator.ale");
}

static PyObject *CreateALE(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"rom_path", "seed", "repeat_action_probability",
                                 "display_screen", "sound", "color_avg",
                                 "frame_skip", "max_num_frames", "max_num_episodes",
                                 "max_num_frames_per_episode", "print_screen",
                                 "system_reset_steps",
                                 "use_environment_distribution",
                                 "num_random_environments", NULL};
  /*
   * Generates and configures the ALE and returns a Python capsule wrapped
   * around the pointer. This capsule is not usable directly by python,
   * but it can be passed to other C++ functions which can unpack and use
   * the pointer. According to python documentation when the
   * capsule passes out of scope in Python, it will call
   * the DeleteALE destructor on the capsule, which will erase the ALE.
   *
   * WARNING: This pointer isn't const because it has to be cast to void in
   * order to feed it to the Python Capsule. Make sure no down-stream redirects
   * the pointer, else the ALE will be lost.
   *
   */

  char *rom_path;
  int seed;
  float repeat_action_probability(0.0);
  int display_screen(0); // int instead of bool because api has problem w/ bool
  int sound(0); // bool
  int color_avg(1); // bool
  int frame_skip(1);
  int max_num_frames(0);
  int max_num_episodes(0);
  int max_num_frames_per_episode(0);
  int print_screen(0); // bool
  int num_reset_steps(4);
  int stochastic_environment(1);
  int num_random_environments(30);

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "si|fiiiiiiiiiii", keyword_list,
                                   &rom_path, &seed, &repeat_action_probability,
                                   &display_screen, &sound, &color_avg,
                                   &frame_skip, &max_num_frames, &max_num_episodes,
                                   &max_num_frames_per_episode, &print_screen,
                                   &num_reset_steps, &stochastic_environment,
                                   &num_random_environments)){
    std::cerr << "Error parsing ALE arguments" << std::endl;
    return NULL;
  }

  if (frame_skip < 1) {
    frame_skip = 1;
    std::cout << "Warning, frame_skip can't be less than 1, setting to 1" << std::endl;
  }

  ALEInterface* ale = new ALEInterface();
  ale->setInt("random_seed", seed);
  ale->setFloat("repeat_action_probability", repeat_action_probability);
  ale->setBool("display_screen", static_cast<bool>(display_screen));
  ale->setBool("sound", static_cast<bool>(sound));
  ale->setBool("print_screen", static_cast<bool>(print_screen));
  ale->setBool("color_averaging", static_cast<bool>(color_avg));
  ale->setInt("frame_skip", frame_skip);
  ale->setInt("max_num_frames", max_num_frames);
  ale->setInt("max_num_episodes", max_num_episodes);
  ale->setInt("max_num_frames_per_episode", max_num_frames_per_episode);
  ale->setInt("system_reset_steps", num_reset_steps);
  ale->setBool("use_environment_distribution", static_cast<bool>(stochastic_environment));
  ale->setInt("num_random_environments", num_random_environments);
  ale->loadROM(rom_path);

  PyObject* ale_capsule = PyCapsule_New(static_cast<void*>(ale),
                              "ale_generator.ale", DeleteALE);
  return ale_capsule;
}

static PyMethodDef ALEMethods[] = {
  { "CreatALE", (PyCFunction) CreateALE, METH_VARARGS | METH_KEYWORDS,
      "Returns a handle to an ALE"},
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef ALEModule = {
  PyModuleDef_HEAD_INIT,
  "ale_generator",
  "Returns a handle to an ALE",
  -1,
  ALEMethods
};

PyMODINIT_FUNC PyInit_ale_generator(void) {
  return PyModule_Create(&ALEModule);
}
