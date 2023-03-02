#ifndef PYBIND_HELPER_H
#define PYBIND_HELPER_H
#include <string>
#include <iostream>
#include <array>

#ifdef USE_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
template <typename T, int T2>
bool operator== (const py::array_t<T, T2>& data, std::vector<size_t>& sizes)
{
	if (data.ndim() != sizes.size()) return false;
	for (int i = 0; i < sizes.size(); i++)
	{
		if (data.shape(i) != sizes[i]) return false;
	}
	return true;
}

template <typename T, int T2>
bool operator!= (const py::array_t<T, T2>& data, std::vector<size_t>& sizes)
{
	return !(data == sizes);
}

template <typename T, int T2>
void print_size(const py::array_t<T, T2>& data)
{
	for (int i = 0; i < data.ndim(); i++)
	{
		if (i != 0) std::cerr << ",";
		std::cerr << data.shape(i);
	}
	std::cerr << std::endl;
}
#endif

template <typename T>
inline std::string get_pybind_class_name(std::string name)
{
	return name;
}

template <>
inline std::string get_pybind_class_name<double>(std::string name)
{
	return name + "Double";
}

template <>
inline std::string get_pybind_class_name<float>(std::string name)
{
	return name + "Float";
}

template <>
inline std::string get_pybind_class_name<long>(std::string name)
{
	return name + "Long";
}

template <>
inline std::string get_pybind_class_name<int>(std::string name)
{
	return name + "Int";
}

template <>
inline std::string get_pybind_class_name<short>(std::string name)
{
	return name + "Short";
}

template <>
inline std::string get_pybind_class_name<char>(std::string name)
{
	return name + "Char";
}

template <>
inline std::string get_pybind_class_name<unsigned long>(std::string name)
{
	return name + "UnsignedLong";
}

template <>
inline std::string get_pybind_class_name<unsigned int>(std::string name)
{
	return name + "UnsignedInt";
}

template <>
inline std::string get_pybind_class_name<unsigned short>(std::string name)
{
	return name + "UnsignedShort";
}

template <>
inline std::string get_pybind_class_name<unsigned char>(std::string name)
{
	return name + "UnsignedChar";
}

#endif

