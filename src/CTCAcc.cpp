#ifdef USE_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef USE_TORCH
#include <torch/torch.h>
#ifdef USE_PYBIND
#include <torch/extension.h>
#endif
#endif

#ifdef USE_OPENCL
#include "OpenCL/CTCOpenCL.h"
#endif
#ifdef USE_CUDA
#include "Cuda/CTCCuda.h"
#endif
#include "PyBindHelper.h"

namespace py = pybind11;

template <typename T>
void get_ctc_class(py::module& m)
{
	py::class_<CTC<T>>(m, get_pybind_class_name<T>("CTC").c_str())
		.def(py::init<>())
#ifdef USE_TORCH
		.def("calcCTCTorch", &CTC<T>::calcCTCTorch)
#endif
		.def("calcCTC", &CTC<T>::calcCTCPy)
		.def("isValid", &CTC<T>::isValid)
		.def("printDeviceInfo", &CTC<T>::printDeviceInfo);
#ifdef USE_OPENCL
	py::class_<CTCOpenCL<T>, CTC<T>>(m, get_pybind_class_name<T>("CTCOpenCL").c_str())
		.def(py::init<bool, bool>(), py::arg("split_forward_backward") = false, py::arg("device_from_stdin") = false);
#endif
#ifdef USE_CUDA
	py::class_<CTCCuda<T>, CTC<T>>(m, get_pybind_class_name<T>("CTCCuda").c_str())
		.def(py::init<bool, bool, bool>(), py::arg("split_forward_backward") = false, py::arg("compile_static_matrix") = true, py::arg("sync_native") = false);
#endif
}

PYBIND11_MODULE(soft_ctc_gpu, m) {
	get_ctc_class<float>(m);
	get_ctc_class<double>(m);
}
#endif

