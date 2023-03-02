#ifndef TENSOR_ND_H
#define TENSOR_ND_H

#include <vector>
#include <string>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

template <typename T, size_t N>
class TensorND
{
public:
	TensorND(T* data, const std::array<size_t, N> &sizes);
	TensorND(const std::array<size_t, N>& sizes);
	TensorND(std::string& filename);
	TensorND(const char* filename);
	TensorND();
	~TensorND(){};
	size_t getElementCount(unsigned int end_level = N, unsigned int start_level = 0);
	bool saveLevels(std::ofstream& stream, size_t offset, unsigned int level, unsigned char col_separator);
	bool loadLevels(std::ifstream& stream, size_t offset, unsigned int level, unsigned char col_separator);
	bool save(std::string &filename);
	bool save(const char *filename);
	bool isValid();
	void resize(const std::array<size_t, N>& sizes);
	std::array<size_t, N> sizes;
	std::vector<T> data;
private:
	bool load(std::string& filename);
	bool valid;
};

template <typename T, size_t N>
TensorND<T, N>::TensorND(const std::array<size_t, N>& sizes)
{
	size_t element_count = 1;
	for (size_t size : sizes)
	{
		element_count *= size;
	}
	this->sizes = sizes;
	this->data.resize(element_count);
	valid = true;
}

template <typename T, size_t N>
TensorND<T, N>::TensorND()
{
	this->sizes.fill(0);
	valid = false;
}

template <typename T, size_t N>
TensorND<T,N>::TensorND(T* data, const std::array<size_t, N>& sizes)
{
	size_t element_count = 1;
	for (size_t size : sizes)
	{
		element_count *= size;
	}
	this->sizes = sizes;
	this->data.resize(element_count);
	std::memcpy(this->data.data(), data, element_count * sizeof(T));
	valid = true;
}

template <typename T, size_t N>
TensorND<T, N>::TensorND(std::string& filename)
{
	load(filename);
}

template <typename T, size_t N>
TensorND<T, N>::TensorND(const char *filename)
{
	std::string s_filename(filename);
	load(s_filename);
}

template <typename T, size_t N>
bool TensorND<T, N>::load(std::string &filename)
{
	valid = false;
	unsigned char col_separator = ';';
	std::ifstream pos_f(filename);

	if (!pos_f.is_open())
	{
		return false;
	}

	/**/
	std::string first_line, second_line;
	if ((!std::getline(pos_f, first_line)) || (!std::getline(pos_f, second_line))) return false;

	std::istringstream first_s(first_line);
	std::istringstream second_s(second_line);

	unsigned int size_count;
	first_s >> size_count;
	if (first_s.bad() || (size_count != N))
	{
		std::cerr << "Error: Invalid tensor format." << std::endl;
		return false;
	}
	for (int size_id = 0; size_id < N; size_id++)
	{
		second_s >> this->sizes[size_id];
		if (size_id != N - 1) second_s >> col_separator;
	}
	if (second_s.bad() || (this->getElementCount() == 0))
	{
		std::cerr << "Warning: Invalid tensor dimensions." << std::endl;
		return false;
	}
	data.resize(this->getElementCount());
	if (!this->loadLevels(pos_f, 0, N - 1, col_separator)) return false;
	pos_f.close();
	valid = true;
	return true;
}

template <typename T, size_t N>
size_t TensorND<T, N>::getElementCount(unsigned int end_level, unsigned int start_level)
{
	if (N == 0) return 0;
	size_t element_count = 1;
	for (unsigned int size_id = start_level; size_id < end_level; size_id++)
	{
		element_count *= sizes[size_id];
	}
	return element_count;
}

template <typename T, size_t N>
bool TensorND<T, N>::saveLevels(std::ofstream& file_stream, size_t offset, unsigned int act_level, unsigned char col_separator)
{
	unsigned int elements_per_act_level = getElementCount(act_level);
	if (act_level == 0)
	{
		for (int i = 0; i < sizes[act_level]; i++)
		{
			file_stream << data[offset + i];
			if (i != sizes[act_level] - 1) file_stream << col_separator;
		}
	}
	else
	{
		for (int i = 0; i < sizes[act_level]; i++)
		{
			if (!saveLevels(file_stream, offset + elements_per_act_level * i, act_level - 1, col_separator)) return false;
			if (i != sizes[act_level] - 1) for (int i = 0; i < act_level; i++) file_stream << std::endl;
		}
	}
	if (file_stream.bad())
	{
		std::cerr << "Error: Cannot save tensor. " << std::endl;
		return false;
	}
	return true;
}

template <typename T, size_t N>
bool TensorND<T, N>::loadLevels(std::ifstream& file_stream, size_t offset, unsigned int act_level, unsigned char col_separator)
{
	unsigned int elements_per_act_level = getElementCount(act_level);
	if (act_level == 0)
	{
		std::string line;
		if (!std::getline(file_stream, line))
		{
			return false;
		}

		std::istringstream line_s(line);
		for (int i = 0; i < sizes[act_level]; i++)
		{
			line_s >> data[offset + i];
			if (i != sizes[act_level] - 1) line_s >> col_separator;
		}
		if (line_s.bad())
		{
			std::cerr << "Error: TensorND<T, N>::loadLevels: invalid data." << std::endl;
			return false;
		}
	}
	else
	{
		for (int i = 0; i < sizes[act_level]; i++)
		{
			if (!loadLevels(file_stream, offset + elements_per_act_level * i, act_level - 1, col_separator)) return false;
			if (i != sizes[act_level] - 1)
				for (int i = 1; i < act_level; i++)
				{
					std::string line;
					if (!std::getline(file_stream, line))
					{
						return false;
					}
				}
		}
	}
	return true;
}


template <typename T, size_t N>
bool TensorND<T, N>::save(std::string& filename)
{
	if (!isValid()) return false;
	unsigned char col_separator = ';';
	std::ofstream pos_f(filename);

	if (!pos_f.is_open())
	{
		return false;
	}

	pos_f << N << std::endl;
	for (int dim_id = 0; dim_id < sizes.size(); dim_id++)
	{
		pos_f << sizes[dim_id];
		if (dim_id == sizes.size() - 1) pos_f << std::endl;
		else pos_f << col_separator;
	}

	if (this->saveLevels(pos_f, 0, N - 1, col_separator)) return false;

	pos_f.close();
	return true;
}

template <typename T, size_t N>
bool TensorND<T, N>::save(const char *filename)
{
	std::string s_filename(filename);
	return this->save(s_filename);
}

template <typename T, size_t N>
bool TensorND<T, N>::isValid()
{
	return valid;
}

template <typename T, size_t N>
void TensorND<T, N>::resize(const std::array<size_t, N>& sizes)
{
	this->sizes = sizes;
	data.resize(this->getElementCount());
	this->valid = (this->getElementCount() != 0);
	
}

#endif
