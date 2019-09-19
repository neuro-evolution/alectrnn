#ifndef ALECTRNN_COMMON_NERVOUS_SYSTEM_HANDLER_H_
#define ALECTRNN_COMMON_NERVOUS_SYSTEM_HANDLER_H_

#include <Python.h>
#include <vector>
#include "parameter_types.hpp"
#include "../common/multi_array.hpp"

PyObject* ConvertFloatVectorToPyFloat32Array(const std::vector<float>& vec);
PyObject* ConvertParameterTypesToPyArray(const std::vector<nervous_system::PARAMETER_TYPE>& par_types);
PyMODINIT_FUNC PyInit_nn_handler(void);

#endif /* ALECTRNN_COMMON_NERVOUS_SYSTEM_HANDLER_H_ */
