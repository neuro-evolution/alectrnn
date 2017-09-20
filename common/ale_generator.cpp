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
#include "ale_generator.h"
#include <ale_interface.hpp>

static void DeleteALE(PyObject *ale_capsule) {
  delete (ALEInterface *)PyCapsule_GetPointer(ale_capsule, "ale_generator.ale");
}

static PyObject *CreateALE(PyObject *self, PyObject *args, PyObject *kwargs) {
  char *keyword_list[] = {"rom", "seed",
      "repeat_action_probability", "display_screen", "sound", "color_avg",
      "frame_skip", "max_num_frames", "max_num_episodes",
      "max_num_frames_per_episode", NULL};
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
   */

  char *rom;//see if switching to string fixes warning
  int seed;
  double repeat_action_probability(0.0);
  bool display_screen(false);
  bool sound(false);
  bool color_avg(true);
  int frame_skip(1);
  int max_num_frames(0);
  int max_num_episodes(0);
  int max_num_frames_per_episode(0);

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "si|dpppiiii", keyword_list,
      &rom, &seed, &repeat_action_probability, &display_screen, &sound,
      &color_avg, &frame_skip, &max_num_frames, &max_num_episodes,
      &max_num_frames_per_episode)){
    return NULL;
  }

  ALEInterface * ale = new ALEInterface(); //static?
  ale->setInt("random_seed", seed);
  ale->setFloat("repeat_action_probability", repeat_action_probability);
  ale->setBool("display_screen", display_screen);
  ale->setBool("sound", sound);
  ale->setBool("color_averaging", color_avg);
  ale->setInt("frame_skip", frame_skip);
  ale->setInt("max_num_frames", max_num_frames);
  ale->setInt("max_num_episodes", max_num_episodes);
  ale->setInt("max_num_frames_per_episode", max_num_frames_per_episode);
  ale->loadROM(rom);

  PyObject* ale_capsule;
  ale_capsule = PyCapsule_New((void *) ale, "ale_generator.ale", DeleteALE);
  return Py_BuildValue("O", ale_capsule);
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
