#include "OclHelper.h"

std::string empty_string = std::string();

bool operator== (const cl_uchar3 &c1, const cl_uchar3 &c2)
{
  return (c1.s[0] == c2.s[0] &&
    c1.s[1] == c2.s[1] &&
    c1.s[2] == c2.s[2]);
}

bool operator== (const cl_float3 &c1, const cl_float3 &c2)
{
  return (c1.s[0] == c2.s[0] &&
    c1.s[1] == c2.s[1] &&
    c1.s[2] == c2.s[2]);
}

bool operator== (const cl_double3 &c1, const cl_double3 &c2)
{
  return (c1.s[0] == c2.s[0] &&
    c1.s[1] == c2.s[1] &&
    c1.s[2] == c2.s[2]);
}

bool operator!= (const cl_uchar3 &c1, const cl_uchar3 &c2)
{
  return !(c1 == c2);
}

bool operator!= (const cl_float3 &c1, const cl_float3 &c2)
{
  return !(c1 == c2);
}

bool operator!= (const cl_double3 &c1, const cl_double3 &c2)
{
  return !(c1 == c2);
}

//Vrati retezec pro opencl error kod
const char *get_error_string(int32_t msg_id)
{
  switch(msg_id)
  {
  case CL_SUCCESS:
    return "Success!";
  case CL_DEVICE_NOT_FOUND:
    return "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE:
    return "Device not available";
  case CL_COMPILER_NOT_AVAILABLE:
    return "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "Memory object allocation failure";
  case CL_OUT_OF_RESOURCES:
    return "Out of resources";
  case CL_OUT_OF_HOST_MEMORY:
    return "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "Profiling information not available";
  case CL_MEM_COPY_OVERLAP:
    return "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "Image format not supported";
  case CL_BUILD_PROGRAM_FAILURE:
    return "Program build failure";
  case CL_MAP_FAILURE:
    return "Map failure";
  case CL_INVALID_VALUE:
    return "Invalid value";
  case CL_INVALID_DEVICE_TYPE:
    return "Invalid device type";
  case CL_INVALID_PLATFORM:
    return "Invalid platform";
  case CL_INVALID_DEVICE:
    return "Invalid device";
  case CL_INVALID_CONTEXT:
    return "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES:
    return "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE:
    return "Invalid command queue";
  case CL_INVALID_HOST_PTR:
    return "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT:
    return "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "Invalid image format descriptor";
  case CL_INVALID_IMAGE_SIZE:
    return "Invalid image size";
  case CL_INVALID_SAMPLER:
    return "Invalid sampler";
  case CL_INVALID_BINARY:
    return "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS:
    return "Invalid build options";
  case CL_INVALID_PROGRAM:
    return "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "Invalid program executable";
  case CL_INVALID_KERNEL_NAME:
    return "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION:
    return "Invalid kernel definition";
  case CL_INVALID_KERNEL:
    return "Invalid kernel";
  case CL_INVALID_ARG_INDEX:
    return "Invalid argument index";
  case CL_INVALID_ARG_VALUE:
    return "Invalid argument value";
  case CL_INVALID_ARG_SIZE:
    return "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS:
    return "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION:
    return "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET:
    return "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "Invalid event wait list";
  case CL_INVALID_EVENT:
    return "Invalid event";
  case CL_INVALID_OPERATION:
    return "Invalid operation";
  case CL_INVALID_GL_OBJECT:
    return "Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE:
    return "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL:
    return "Invalid mip-map level";
  default:
    return "Unknown";
  }
}

bool oclPrintErrorExit(int32_t err_num, const char *text)
{
  if((err_num) < 0 && (err_num != CL_DEVICE_NOT_FOUND))
  {
    std::cerr << "Error: " << text << ": (" << err_num << ") " << get_error_string(err_num) << std::endl;
    std::string input_data;
    std::cin >> input_data;
    exit(1);
    return false;
  }
  return true;
}

bool oclPrintError(int32_t err_num, const char *text, std::ostream *error_stream)
{
  if((err_num != CL_SUCCESS) && (error_stream != NULL)) *error_stream << "Error: " << text << ": (" << err_num << ") " << get_error_string(err_num) << std::endl;
  return (err_num == CL_SUCCESS);
}

bool oclPrintInfo(int32_t err_num, const char *text, std::ostream *error_stream)
{
  if((err_num != CL_SUCCESS) && (error_stream != NULL)) *error_stream << "Info: " << text << ": (" << err_num << ") " << get_error_string(err_num) << std::endl;
  return (err_num == CL_SUCCESS);
}

bool read_file(const std::string &filename, std::string &out_data)
{
  std::ifstream file(filename.c_str(), std::fstream::binary | std::fstream::in);
  if(!file.is_open()) return false;

  file.seekg(0, std::ios_base::end);
  size_t file_length = (size_t)file.tellg();
  file.seekg(0, std::ios_base::beg);

  out_data.resize(file_length);

  file.read(&out_data[0], file_length);

  if(file.fail()) return false;

  file.close();
  return true;
}

std::string read_file(const std::string &file_name)
{
  std::string file_data;
  return read_file(file_name, file_data) ? file_data : empty_string;
}

bool oclCreateProgramWithSource(cl_program &program, cl_context context, cl_device_id device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream)
{
  int32_t err_num;
  std::string src_data;
  if(!read_file(kernel_file_name, src_data)) { if(error_stream != NULL) *error_stream << "Error: Cannot read kernel file.\n"; return false; }
  const char *src_data_c = src_data.c_str();
  program = clCreateProgramWithSource(context, 1, &src_data_c, NULL, &err_num);
  if(!oclPrintError(err_num, "clCreateProgramWithSource", error_stream)) return false;
  if(!oclPrintError(clBuildProgram(program, 1, &device, program_defines.c_str(), NULL, NULL), "clBuildProgram", error_stream))
  {
    size_t program_size;
    if(!oclPrintError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &program_size), "clGetProgramBuildInfo", error_stream)) return false;
    if(program_size == 0) return false;
    char *data = (char *)malloc(program_size);
    if(!oclPrintError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, program_size, data, NULL), "clGetProgramBuildInfo", error_stream)) { free(data); return false; }
    if(error_stream != NULL) *error_stream << "Build log:\n" << data << "\n";
    free(data);
    return false;
  }
  return true;
}

bool oclCreateProgramWithSource(cl::Program &program, const cl::Context &context, const cl::Device &device, const std::string &program_defines, const std::string &program_file_name, std::ostream *error_stream)
{
	int32_t err_msg, err_msg_log;
	std::string p_source;
	if (!read_file(program_file_name, p_source)) { if (error_stream != NULL) *error_stream << "Error: Cannot read kernel file.\n"; return false; }
	cl::Program::Sources p_sources;
	p_sources.emplace_back(p_source.c_str(), 0);

	program = cl::Program(context, p_sources, &err_msg);
	if (!oclPrintError(err_msg, "BoostDetector: cl::Program")) return false;

	err_msg = program.build(std::vector<cl::Device>(1, device), program_defines.c_str(), NULL, NULL);

#ifndef DEBUG_BUILD_LOG
	if (err_msg != CL_BUILD_SUCCESS)
#endif
	{
		std::cerr << "Build parameters: " << program_defines << std::endl;
		std::cerr << "Build log " << program_file_name << " : " << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err_msg_log) << std::endl;
		oclPrintErrorExit(err_msg_log, "BoostDetector: cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG> p_detector");
	}
	oclPrintErrorExit(err_msg, "BoostDetector: cl::Program::build p_detector");

	return true;
}

size_t getMaximumGpuThreads(const cl::Device &device)
{
  uint32_t max_compute_units;
  oclPrintErrorExit(device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &max_compute_units), "Error: Canot get CL_DEVICE_MAX_COMPUTE_UNITS device info.");
  return max_compute_units * 2048;
}

#ifdef COMPUTE_CACHE_ENABLED
#ifdef BOOST_ENABLE
bool createBinaryConfig(std::string &binary_config, cl_device_id device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream)
{
  size_t data_len;
  size_t act_offset;
  std::string cl_info;
  std::ostringstream out_stream;
  std::vector<cl_device_info> v_device_info = {CL_DEVICE_VENDOR, CL_DEVICE_NAME, CL_DEVICE_VERSION, CL_DRIVER_VERSION};

  // check existence of file
  boost::filesystem::path kernel_file_path(kernel_file_name);
  if(!boost::filesystem::is_regular_file(kernel_file_path)) return false;

  // append driver info
  for(int i = 0; i < v_device_info.size(); i++)
  {
    if(!oclPrintError(clGetDeviceInfo(device, v_device_info[i], NULL, NULL, &data_len), "clGetDeviceInfo ", error_stream)) continue;
    if(data_len == 0) continue;
    act_offset = cl_info.size();
    cl_info.resize(cl_info.size() + data_len);
    if(!oclPrintError(clGetDeviceInfo(device, v_device_info[i], data_len, &cl_info[act_offset], NULL), "clGetDeviceInfo ", error_stream)) continue;
    cl_info[cl_info.size() - 1] = ';';
  }
  out_stream << cl_info;

  // append file info
  out_stream << kernel_file_name;
  out_stream << ";";
  out_stream << boost::filesystem::last_write_time(kernel_file_path);
  out_stream << ";";
  out_stream << boost::filesystem::file_size(kernel_file_path);
  out_stream << ";";
  out_stream << program_defines;
  binary_config = out_stream.str();
  return true;
}

std::string createStringHash(std::string &in_str)
{
  std::hash<std::string> std_hash;
  return boost::lexical_cast<std::string>(std_hash(in_str));
}

bool getBinaryId(unsigned int &binary_id, const std::string &bin_config, const std::string &hash_dir_str, std::ostream *error_stream)
{
  unsigned int i;
  std::string hash_data;
  for(i = 0; true; i++)
  {
    boost::filesystem::path hash_text_path(hash_dir_str);
    hash_text_path.append(boost::lexical_cast<std::string>(i) + ".txt");
    if(!boost::filesystem::is_regular_file(hash_text_path)) break;
    if(!read_file(hash_text_path.generic_string(), hash_data)) { if(error_stream != NULL) *error_stream << "Error: Cannot read file \"" << hash_text_path.generic_string() << "\"\n"; break; }
    if(bin_config != hash_data) continue;
    binary_id = i;
    return true;
  }
  binary_id = i;
  return false;
}

bool oclSaveBinary(const cl_program &program, const std::string &bin_config, const boost::filesystem::path &hash_dir_path, const unsigned int binary_id, std::ostream *error_stream)
{
  size_t binary_size;
  if(!oclPrintError(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL), "clGetProgramInfo", error_stream)) return false;
  if(binary_size == 0) { if(error_stream != NULL) *error_stream << "Error: clGetProgramInfo zero value\n"; return false; }
  unsigned char *binary = (unsigned char *)malloc(binary_size + 1);
  if(!oclPrintError(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL), "clGetProgramInfo", error_stream)) { free(binary); return false; }

  boost::filesystem::create_directories(hash_dir_path);

  // cache the binary
  boost::filesystem::path hash_file_path = hash_dir_path;
  hash_file_path.append(boost::lexical_cast<std::string>(binary_id) + ".txt");

  std::ofstream hash_text_file(hash_file_path.generic_string().c_str(), std::ios::out);
  if(hash_text_file.bad()) { if(error_stream != NULL) *error_stream << "Error: Cannot open binary info file\n"; return false; }
  hash_text_file << bin_config;
  hash_text_file.close();

  boost::filesystem::path hash_bin_path = hash_dir_path;
  hash_bin_path.append(boost::lexical_cast<std::string>(binary_id) + ".bin");

  std::ofstream hash_bin_file(hash_bin_path.generic_string().c_str(), std::ios::out | std::ios::binary);
  if(hash_text_file.bad()) { if(error_stream != NULL) *error_stream << "Error: Cannot open binary file\n"; return false; }
  hash_bin_file.write((char *)binary, binary_size);
  hash_bin_file.close();
  free(binary);
  return true;
}

bool oclCreateProgramWithBinary(cl_program &program, cl_context context, cl_device_id device, const std::string &program_defines, const std::string &binary_data, std::ostream *error_stream)
{
  int32_t err_num;
  int32_t bin_status;
  size_t binary_size = binary_data.size();
  const unsigned char *binary = (const unsigned char *)binary_data.data();
  program = clCreateProgramWithBinary(context, 1, &device, &binary_size, &binary, &bin_status, &err_num);
  if((!oclPrintError(err_num, "clCreateProgramWithBinary", error_stream)) ||
    (!oclPrintError(bin_status, "clCreateProgramWithBinary", error_stream))) return false;
  cl_program_binary_type bin_type;
  if(!oclPrintError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BINARY_TYPE, sizeof(bin_type), &bin_type, NULL), "clGetProgramBuildInfo", error_stream)) return false;
  if(!oclPrintError(clBuildProgram(program, 1, &device, program_defines.c_str(), NULL, NULL), "clBuildProgram", error_stream))
  {
    size_t program_size;
    if(!oclPrintError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &program_size), "clGetProgramBuildInfo", error_stream)) return false;
    if(program_size == 0) return false;
    char *data = (char *)malloc(program_size);
    if(!oclPrintError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, program_size, data, NULL), "clGetProgramBuildInfo", error_stream)) { free(data); return false; }
    if(error_stream != NULL) *error_stream << "Build log:\n" << data << "\n";
    free(data);
    return false;
  }
  return true;
}

bool oclLoadBinary(std::string &binary_data, const boost::filesystem::path &hash_dir_path, const unsigned int binary_id, std::ostream *error_stream)
{
  boost::filesystem::path hash_bin_path = hash_dir_path;
  hash_bin_path.append(boost::lexical_cast<std::string>(binary_id) + ".bin");
  if(!read_file(hash_bin_path.generic_string(), binary_data)) { if(error_stream != NULL) *error_stream << "Error: Cannot read binary file.\n"; return false; }
  return true;
}

e_ocl_compile compileOpenclSource(cl_program &program, cl_context context, cl_device_id device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream)
{
  std::string bin_config;
  if(!createBinaryConfig(bin_config, device, program_defines, kernel_file_name, error_stream)) return OCL_COMPILE_FAILED;

  // get hash dir
  boost::filesystem::path hash_dir_path(compute_cache_dir);
  hash_dir_path.append(createStringHash(bin_config));

  unsigned int binary_id;
  std::string binary_data;
  if((!getBinaryId(binary_id, bin_config, hash_dir_path.generic_string(), error_stream)) ||
    (!oclLoadBinary(binary_data, hash_dir_path, binary_id, error_stream)) ||
    (!oclCreateProgramWithBinary(program, context, device, program_defines, binary_data, error_stream)))
  {
    if(!oclCreateProgramWithSource(program, context, device, program_defines, kernel_file_name, error_stream)) return OCL_COMPILE_FAILED;
    if(!oclSaveBinary(program, bin_config, hash_dir_path, binary_id, error_stream)) return OCL_COMPILE_FAILED;
    return OCL_COMPILE_COMPILED;
  }

  return OCL_COMPILE_CACHED;
}
#else
bool createBinaryConfig(std::string &binary_config, const cl::Device device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream)
{
	std::ostringstream out_stream;

	// check existence of file
	std::filesystem::path kernel_file_path(kernel_file_name);
	if (!std::filesystem::is_regular_file(kernel_file_path)) return false;

	// append file info
	out_stream << device.getInfo<CL_DEVICE_VENDOR>() << ";";
	out_stream << device.getInfo<CL_DEVICE_NAME>() << ";";
	out_stream << device.getInfo<CL_DEVICE_VERSION>() << ";";
	out_stream << device.getInfo<CL_DRIVER_VERSION>() << ";";
	out_stream << kernel_file_name << ";";
	out_stream << std::filesystem::last_write_time(kernel_file_path).time_since_epoch().count() << ";";
	out_stream << std::filesystem::file_size(kernel_file_path) << ";";
	out_stream << program_defines;
	binary_config = out_stream.str();
	return true;
}

std::string createStringHash(std::string &in_str)
{
	std::hash<std::string> std_hash;
	return std::to_string(std_hash(in_str));
}

bool getBinaryId(unsigned int &binary_id, const std::string &bin_config, const std::string &hash_dir_str, std::ostream *error_stream)
{
	unsigned int i;
	std::string hash_data;
	for (i = 0; true; i++)
	{
		std::filesystem::path hash_text_path(hash_dir_str);
		hash_text_path.append(std::to_string(i) + ".txt");
		if (!std::filesystem::is_regular_file(hash_text_path)) break;
		if (!read_file(hash_text_path.generic_string(), hash_data)) { if (error_stream != NULL) *error_stream << "Error: Cannot read file \"" << hash_text_path.generic_string() << "\"\n"; break; }
		if (bin_config != hash_data) continue;
		binary_id = i;
		return true;
	}
	binary_id = i;
	return false;
}

bool oclSaveBinary(const cl::Program &program, const std::string &bin_config, const std::filesystem::path &hash_dir_path, const unsigned int binary_id, std::ostream *error_stream)
{
	int32_t err_msg;
	std::vector<size_t> binary_sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>(&err_msg);
	oclPrintErrorExit(err_msg, "BoostDetector: cl::Program::getInfo CL_PROGRAM_BINARY_SIZES");
	std::vector<char *> binary_data = program.getInfo<CL_PROGRAM_BINARIES>();
	oclPrintErrorExit(err_msg, "BoostDetector: cl::Program::getInfo CL_PROGRAM_BINARIES");
	if ((binary_data.size() != binary_sizes.size()) || (binary_data.size() != 1)) { if (error_stream != NULL) *error_stream << "BoostDetector: cl::Program invalid number binaries"; return false; }

	std::filesystem::create_directories(hash_dir_path);

	// cache the binary
	std::filesystem::path hash_file_path = hash_dir_path;
	hash_file_path.append(std::to_string(binary_id) + ".txt");

	std::ofstream hash_text_file(hash_file_path.generic_string().c_str(), std::ios::out);
	if (hash_text_file.bad()) { if (error_stream != NULL) *error_stream << "Error: Cannot open binary info file\n"; return false; }
	hash_text_file << bin_config;
	hash_text_file.close();

	std::filesystem::path hash_bin_path = hash_dir_path;
	hash_bin_path.append(std::to_string(binary_id) + ".bin");

	std::ofstream hash_bin_file(hash_bin_path.generic_string().c_str(), std::ios::out | std::ios::binary);
	if (hash_text_file.bad()) { if (error_stream != NULL) *error_stream << "Error: Cannot open binary file\n"; return false; }
	hash_bin_file.write(binary_data[0], binary_sizes[0]);
	hash_bin_file.close();

	for (int bin_id = 0; bin_id < binary_data.size(); bin_id++)
	{
		delete binary_data[bin_id];
	}
	return true;
}

bool oclCreateProgramWithBinary(cl::Program &program, const cl::Context &context, const cl::Device &device, const std::string &program_defines, const std::string &binary_data, std::ostream *error_stream)
{
	int32_t err_msg, err_msg_log;
	cl::Program::Binaries bins;
	std::vector<int32_t> bin_status;
	bins.push_back(std::pair<const void *, size_t>(binary_data.data(), binary_data.size()));
	program = cl::Program(context, std::vector<cl::Device>(1, device), bins, &bin_status, &err_msg);
	if ((!oclPrintError(err_msg, "clCreateProgramWithBinary")) ||
		(!oclPrintError(bin_status[0], "clCreateProgramWithBinary"))) return false;

	cl_program_binary_type bin_type = program.getBuildInfo<CL_PROGRAM_BINARY_TYPE>(device, &err_msg);
	if (!oclPrintError(err_msg, "clGetProgramBuildInfo")) return false;

	err_msg = program.build(std::vector<cl::Device>(1, device), program_defines.c_str(), NULL, NULL);

#ifndef DEBUG_BUILD_LOG
	if (err_msg != CL_BUILD_SUCCESS)
#endif
	{
		std::cerr << "Build parameters: " << program_defines << std::endl;
		std::cerr << "Build log : " << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err_msg_log) << std::endl;
		oclPrintErrorExit(err_msg_log, "BoostDetector: cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG>");
	}
	return oclPrintErrorExit(err_msg, "BoostDetector: cl::Program::build");
}

bool oclLoadBinary(std::string &binary_data, const std::filesystem::path &hash_dir_path, const unsigned int binary_id, std::ostream *error_stream)
{
	std::filesystem::path hash_bin_path = hash_dir_path;
	hash_bin_path.append(std::to_string(binary_id) + ".bin");
	if (!read_file(hash_bin_path.generic_string(), binary_data)) { if (error_stream != NULL) *error_stream << "Error: Cannot read binary file.\n"; return false; }
	return true;
}

e_ocl_compile compileOpenclSource(cl::Program &program, const cl::Context &context, const cl::Device &device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream)
{
	std::string bin_config;
	if (!createBinaryConfig(bin_config, device, program_defines, kernel_file_name, error_stream)) return OCL_COMPILE_FAILED;

	// get hash dir
	std::filesystem::path hash_dir_path(compute_cache_dir);
	hash_dir_path.append(createStringHash(bin_config));

	unsigned int binary_id;
	std::string binary_data;
	if ((!getBinaryId(binary_id, bin_config, hash_dir_path.generic_string(), error_stream)) ||
		(!oclLoadBinary(binary_data, hash_dir_path, binary_id, error_stream)) ||
		(!oclCreateProgramWithBinary(program, context, device, program_defines, binary_data, error_stream)))
	{
		if (!oclCreateProgramWithSource(program, context, device, program_defines, kernel_file_name, error_stream)) return OCL_COMPILE_FAILED;
		if (!oclSaveBinary(program, bin_config, hash_dir_path, binary_id, error_stream)) return OCL_COMPILE_FAILED;
		return OCL_COMPILE_COMPILED;
	}

	return OCL_COMPILE_CACHED;
}
#endif
#else
e_ocl_compile compileOpenclSource(cl_program &program, cl_context context, cl_device_id device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream)
{
  if(!oclCreateProgramWithSource(program, context, device, program_defines, kernel_file_name, error_stream)) return OCL_COMPILE_FAILED;
  return OCL_COMPILE_COMPILED;
}
e_ocl_compile compileOpenclSource(cl::Program &program, const cl::Context &context, const cl::Device &device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream)
{
	if(!oclCreateProgramWithSource(program, context, device, program_defines, kernel_file_name, error_stream)) return OCL_COMPILE_FAILED;
	return OCL_COMPILE_CACHED;
}
#endif

double getEventTime(cl_event i_event)
{
  uint64_t time_from, time_to;
  clGetEventProfilingInfo(i_event, CL_PROFILING_COMMAND_START, sizeof(uint64_t), &time_from, NULL);
  clGetEventProfilingInfo(i_event, CL_PROFILING_COMMAND_END, sizeof(uint64_t), &time_to, NULL);
  return (time_to - time_from) / 1000000000.0;
}

double getEventTime(cl::Event &event)
{
  return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000000.0;
}

std::string oclGetDeviceSpecificDefines(const cl::Device &device)
{
	int32_t err_msg, err_msg2;
  bool nvidia = false;
  std::string extensions;
  std::string platform;
  //std::string vendor;
  std::string name;
  cl_device_type device_type;

  e_device_type ocl_device_type;

  uint32_t warp_size = 1;

  // get device attributes
  oclPrintErrorExit(device.getInfo(CL_DEVICE_EXTENSIONS, &extensions), "GpuDetector: cl::Device::getInfo CL_DEVICE_EXTENSIONS");
  //oclPrintErrorExit(device.getInfo(CL_DEVICE_VENDOR, &vendor), "GpuDetector: cl::Device::getInfo CL_DEVICE_VENDOR");
  oclPrintErrorExit(device.getInfo(CL_DEVICE_NAME, &name), "GpuDetector: cl::Device::getInfo CL_DEVICE_NAME");
  oclPrintErrorExit(device.getInfo(CL_DEVICE_TYPE, &device_type), "GpuDetector: cl::Device::getInfo CL_DEVICE_TYPE");
  platform = cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>(&err_msg)).getInfo<CL_PLATFORM_NAME>(&err_msg2);
  oclPrintErrorExit(err_msg, "GpuDetector: cl::Device::getInfo CL_DEVICE_PLATFORM");
  oclPrintErrorExit(err_msg2, "GpuDetector: cl::Platform::getInfo CL_PLATFORM_NAME");

  // convert names to upper case
  std::transform(extensions.begin(), extensions.end(), extensions.begin(), toupper);
  std::transform(platform.begin(), platform.end(), platform.begin(), toupper);
  std::transform(name.begin(), name.end(), name.begin(), toupper);

  // set device type
  if(platform.find("NVIDIA") != std::string::npos)
  {
    if(device_type == CL_DEVICE_TYPE_GPU) ocl_device_type = DEVICE_TYPE_NVIDIA_GPU;
    else if(device_type == CL_DEVICE_TYPE_CPU) ocl_device_type = DEVICE_TYPE_NVIDIA_CPU;
    else ocl_device_type = DEVICE_TYPE_OTHER;
  }
  else if(platform.find("AMD") != std::string::npos)
  {
    if(device_type == CL_DEVICE_TYPE_GPU) ocl_device_type = DEVICE_TYPE_AMD_GPU;
    else if(device_type == CL_DEVICE_TYPE_CPU) ocl_device_type = DEVICE_TYPE_AMD_CPU;
    else ocl_device_type = DEVICE_TYPE_OTHER;
  }
  else if(platform.find("INTEL") != std::string::npos)
  {
    if(device_type == CL_DEVICE_TYPE_GPU) ocl_device_type = DEVICE_TYPE_INTEL_GPU;
    else if(device_type == CL_DEVICE_TYPE_CPU) ocl_device_type = DEVICE_TYPE_INTEL_CPU;
    else ocl_device_type = DEVICE_TYPE_OTHER;
  }
  else
  {
    ocl_device_type = DEVICE_TYPE_OTHER;
  }

  std::ostringstream str("");
  str << " -D DEVICE_TYPE=" << ocl_device_type;

  // get specific vendor info
  switch(ocl_device_type)
  {
  case DEVICE_TYPE_NVIDIA_GPU:
    if(extensions.find("CL_NV_DEVICE_ATTRIBUTE_QUERY") != std::string::npos)
    {
      uint32_t nvidia_sm_major = 0;
      uint32_t nvidia_sm_minor = 0;
      oclPrintErrorExit(device.getInfo(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, &nvidia_sm_major), "GpuDetector: cl::Device::getInfo CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV");
      oclPrintErrorExit(device.getInfo(CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, &nvidia_sm_minor), "GpuDetector: cl::Device::getInfo CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV");
      oclPrintErrorExit(device.getInfo(CL_DEVICE_WARP_SIZE_NV, &warp_size), "GpuDetector: cl::Device::getInfo CL_DEVICE_WARP_SIZE_NV");
      str << " -D CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV=" << nvidia_sm_major
        << " -D CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV=" << nvidia_sm_minor;
    }
    break;
  case DEVICE_TYPE_AMD_GPU:
    if(extensions.find("CL_AMD_DEVICE_ATTRIBUTE_QUERY"))
    {
      oclPrintErrorExit(device.getInfo(CL_DEVICE_WAVEFRONT_WIDTH_AMD, &warp_size), "GpuDetector: cl::Device::getInfo CL_DEVICE_WAVEFRONT_WIDTH_AMD");
    }
    break;
  }
  str << " -D WARP_SIZE=" << warp_size;
  return str.str();
}


KernelRange::KernelRange(uint32_t x, uint32_t y, uint32_t z)
{
  this->x = x;
  this->y = y;
  this->z = z;
}

KernelRange::KernelRange(uint32_t x, uint32_t y, uint32_t z, KernelRange &local)
{
  this->x = x;
  this->y = y;
  this->z = z;
  alignRangeTo(local);
}
bool KernelRange::isAligned(KernelRange &local)
{
  return ((x % local.x == 0) && (y % local.y == 0) && (z % local.z == 0));
}

void KernelRange::ceilRangeDivBy(KernelRange &local)
{
  x = ceilDivBy(x, local.x);
  y = ceilDivBy(y, local.y);
  z = ceilDivBy(z, local.z);
}

bool KernelRange::isValid()
{
  return (x > 0) && (y > 0) && (z > 0);
}
void KernelRange::alignRangeTo(KernelRange &local)
{
  x = alignTo(x, local.x);
  y = alignTo(y, local.y);
  z = alignTo(z, local.z);
}

cl::NDRange KernelRange::toNDRange()
{
  return cl::NDRange(x, y, z);
}

KernelRange KernelRange::operator +(const KernelRange &data)
{
  return KernelRange(x + data.x, y + data.y, z + data.z);
}

KernelRange KernelRange::operator -(const KernelRange &data)
{
  return KernelRange(x - data.x, y - data.y, z - data.z);
}

bool KernelRange::operator ==(const KernelRange &data)
{
  return (data.x == x) && (data.y == y) && (data.z == z);
}

uint32_t KernelRange::getFlatSize()
{
  return x*y*z;
}

bool KernelRange::isZero()
{
  return (x == 0) && (y == 0) && (z == 0);
}

KernelRange KernelRange::getOverlappingSizeArea(const KernelRange &limit)
{
  return KernelRange((x > limit.x) ? (x - limit.x) : 0,
    (y > limit.y) ? (y - limit.y) : 0,
    (z > limit.z) ? (z - limit.z) : 0);
}

void KernelRange::zeroOverlappingArea(const KernelRange &limit)
{
  if(x >= limit.x) x = 0;
  else y -= 1;
  if(y >= limit.y) y = 0;
  else z -= 1;
  if(z >= limit.z) z = 0;
}


KernelRangeData::KernelRangeData(KernelRange &total_size, KernelRange &local_size, uint32_t max_size)
{
  this->total_size = total_size;
  this->total_size.alignRangeTo(local_size);
  this->local = local_size;

  this->max_kernel_size = getMaxKernelRange(max_size);

  this->init();
}

bool KernelRangeData::isBegin()
{
  return act_start == KernelRange(0, 0, 0);
}

void KernelRangeData::init()
{
  act_start = KernelRange(0, 0, 0);

  offset = act_start;
  global = max_kernel_size;
}

void KernelRangeData::next()
{
  act_start = act_start + max_kernel_size;
  act_start.zeroOverlappingArea(total_size);
  offset = act_start;

  global = (max_kernel_size - (act_start + max_kernel_size).getOverlappingSizeArea(total_size));
}

bool KernelRangeData::isValid()
{
  return max_kernel_size.isValid();
}

KernelRange KernelRangeData::getMaxKernelRange(uint32_t max_size)
{
  int local_flat_size = local.getFlatSize();
  if((local_flat_size > max_size) || (!total_size.isValid()) || (!local.isValid())) return KernelRange(0, 0, 0);
  if(max_size / (local.y * local.z * total_size.x) == 0) return KernelRange(alignTo(max_size / (local.y * local.z), local.x), local.y, local.z);
  if(max_size / (local.z * total_size.y * total_size.x) == 0) return KernelRange(total_size.x, alignTo(max_size / (total_size.x * local.z), local.y), local.z);
  if(max_size / (total_size.z * total_size.y * total_size.x) == 0) return KernelRange(total_size.x, total_size.y, alignTo(max_size / (total_size.x * total_size.y), local.z));
  return KernelRange(total_size.x, total_size.y, total_size.z);
}





OclKernel::OclKernel(const cl::Program& program, const char* name, const cl::Device& device, int const_shared, int shared_per_thread, int min_shared_per_group, int force_align_to_size): cl::Kernel(program, name, &err_msg)
{
  args_count = 0;
  oclPrintErrorExit(err_msg, (std::string("create kernel ") + name).c_str());
  setMaxGroupSize(device, const_shared, shared_per_thread, min_shared_per_group, force_align_to_size);
}


KernelRange OclKernel::getGroupSize()
{
  return group_size;
}

uint32_t OclKernel::getDynamicMemSize()
{
  return dynamic_mem_size;
}


void OclKernel::setMaxGroupSize(const cl::Device& device, int const_shared, int shared_per_thread, int min_shared_per_group, int force_align_to_size)
{
  uint64_t device_local_mem;
  uint64_t static_local_mem_size;
  size_t max_work_group_size;
  size_t warp_size;
  oclPrintErrorExit(device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &device_local_mem), "cl::Device::getInfo CL_DEVICE_LOCAL_MEM_SIZE");
  oclPrintErrorExit(this->getWorkGroupInfo(device, CL_KERNEL_LOCAL_MEM_SIZE, &static_local_mem_size), "cl::Kernel::getWorkGroupInfo CL_KERNEL_LOCAL_MEM_SIZE");
  oclPrintErrorExit(this->getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &max_work_group_size), "cl::Kernel::getWorkGroupInfo CL_KERNEL_WORK_GROUP_SIZE");
  oclPrintErrorExit(this->getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &warp_size), "cl::Kernel::getWorkGroupInfo CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
  warp_size = alignTo(warp_size, (size_t)force_align_to_size); // align to multiple force_align_to_size
  max_work_group_size = (max_work_group_size / warp_size) * warp_size; // floor to warp size

  if((int)device_local_mem - (int)static_local_mem_size - const_shared - min_shared_per_group < 0)
  {
    this->dynamic_mem_size = 0;
    this->max_group_size = 0;
  }
  else if(shared_per_thread == 0)
  {
    this->dynamic_mem_size = const_shared + min_shared_per_group;
    this->max_group_size = max_work_group_size;
  }
  else
  {
    this->max_group_size = std::min((((device_local_mem - static_local_mem_size - const_shared) / shared_per_thread) / warp_size) * warp_size, (uint64_t) max_work_group_size);
    this->dynamic_mem_size = std::max(const_shared + max_group_size * shared_per_thread, (unsigned int)const_shared + min_shared_per_group);
  }
  if(max_group_size == 0)
  {
    oclPrintErrorExit(CL_INVALID_DEVICE, (std::string("Insufficeient resources run kernel:") + this->getInfo<CL_KERNEL_FUNCTION_NAME>()).c_str());
  }
  if(force_align_to_size <= 1) group_size = KernelRange(max_group_size, 1, 1);
  else group_size = KernelRange(force_align_to_size, max_group_size / force_align_to_size, 1);
}

std::vector<cl::Device> oclGetDevices(cl_device_type device_type)
{
	std::vector<cl::Device> devices;

  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> platform_devices;

  oclPrintErrorExit(cl::Platform::get(&platforms), "cl::Platform::get");

  for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it)
  {
    platform_devices.clear();
    oclPrintErrorExit(plat_it->getDevices(device_type, &platform_devices), "getDevices");
    devices.insert(devices.end(), platform_devices.begin(), platform_devices.end());
  }
  return devices;
}

cl::Device oclGetDeviceFromParameters(cl_device_type device_type, const std::string &device_vendor, const std::string &device_name)
{
  bool filter_by_vendor = (device_vendor != empty_string);
  bool filter_by_name = (device_name != empty_string);
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> platform_devices;

  oclPrintErrorExit(cl::Platform::get(&platforms), "cl::Platform::get");

  for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it)
  {
    if((!filter_by_vendor) || (plat_it->getInfo<CL_PLATFORM_VENDOR>().find(device_vendor) != std::string::npos))
    {
      platform_devices.clear();
      oclPrintErrorExit(plat_it->getDevices(device_type, &platform_devices), "getDevices");
      for(int i = 0; i < platform_devices.size(); i++)
      {
        if((!filter_by_name) || (platform_devices[i].getInfo<CL_DEVICE_NAME>().find(device_name) != std::string::npos)) return platform_devices[i];
      }
    }
  }
  if(filter_by_vendor)
  {
    std::cerr << "Warning: OpenCL device with platform \"" << device_vendor << "\" is not found" << std::endl << "Info: Trying to select other device platform." << std::endl;;
    for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it)
    {
      platform_devices.clear();
      oclPrintErrorExit(plat_it->getDevices(device_type, &platform_devices), "getDevices");
      for(int i = 0; i < platform_devices.size(); i++)
      {
        if((!filter_by_name) || (platform_devices[i].getInfo<CL_DEVICE_NAME>().find(device_name) != std::string::npos)) return platform_devices[i];
      }
    }
  }
  if(filter_by_name)
  {
    std::cerr << "Warning: OpenCL device with name \"" << device_name << "\" is not found" << std::endl << "Info: Trying to select other device name." << std::endl;;
    for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it)
    {
      platform_devices.clear();
      oclPrintErrorExit(plat_it->getDevices(device_type, &platform_devices), "getDevices");
      for(int i = 0; i < platform_devices.size(); i++)
      {
        return platform_devices[i];
      }
    }
  }
  if(device_type != CL_DEVICE_TYPE_ALL)
  {
    std::cerr << "Warning: Device with selected device type is not found." << std::endl << "Info: Trying to select other device types." << std::endl;
    for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it)
    {
      platform_devices.clear();
      oclPrintErrorExit(plat_it->getDevices(CL_DEVICE_TYPE_ALL, &platform_devices), "getDevices");
      for(int i = 0; i < platform_devices.size(); i++)
      {
        return platform_devices[i];
      }
    }
  }
  return cl::Device();
}

void oclPrintDevices(const std::vector<cl::Device> &devices)
{
	for (std::vector<cl::Device>::const_iterator dev_it = devices.begin(); dev_it != devices.end(); ++dev_it)
	{
		std::cerr << (dev_it - devices.begin()) << std::endl;
		oclPrintDeviceInfo(*dev_it, std::cerr);
	}
}

std::string printDeviceType(cl_device_type type)
{
	switch (type)
	{
	case CL_DEVICE_TYPE_GPU:
		return std::string("GPU");
		break;
	case CL_DEVICE_TYPE_CPU:
		return std::string("CPU");
		break;
	case CL_DEVICE_TYPE_ACCELERATOR:
		return std::string("ACCELERATOR");
		break;
	default:
		return std::string("OTHER");
		break;
	}
}


cl::Device oclSelectDeviceByStdin(const std::vector<cl::Device> &devices)
{
  if(devices.size() == 0) return cl::Device();
  int32_t err_msg, err_msg2, err_msg3, err_msg4;

  // Get devices count
  std::cerr << std::endl << "Devices:" << std::endl;
  for(std::vector<cl::Device>::const_iterator dev_it = devices.begin(); dev_it != devices.end(); ++dev_it)
  {
	  std::cerr << " " << (dev_it - devices.begin()) << ": "
				<< printDeviceType(dev_it->getInfo<CL_DEVICE_TYPE>(&err_msg4)) << " - "
				<< cl::Platform(dev_it->getInfo<CL_DEVICE_PLATFORM>(&err_msg)).getInfo<CL_PLATFORM_NAME>(&err_msg2) << " - "
				<< dev_it->getInfo<CL_DEVICE_NAME>(&err_msg3) << std::endl;
		oclPrintErrorExit(err_msg, "cl::Device.getInfo<CL_DEVICE_PLATFORM>");
		oclPrintErrorExit(err_msg2, "cl::Platform.getInfo<CL_PLATFORM_NAME>");
		oclPrintErrorExit(err_msg3, "cl::Device.getInfo<CL_DEVICE_NAME>");
		oclPrintErrorExit(err_msg3, "cl::Device.getInfo<CL_DEVICE_TYPE>");
  }

  cl::Device act_device;
  int user_device_id;
  while(1)
  {
   std::cerr << std::endl << "Select device id:" << std::endl;
	std::cin >> user_device_id;
    if(user_device_id < devices.size())
    {
      act_device = devices[user_device_id];
      break;
    }
    else
    {
      std::cerr << std::endl << "Selected device id is out of range. Right range is <" << 0 << ", " << (devices.size() - 1) << ">" << std::endl;
    }
  }

  std::cerr << "Selected device name: " << act_device.getInfo<CL_DEVICE_NAME>(&err_msg) << " ." << std::endl;
  oclPrintErrorExit(err_msg, "cl::Device::getInfo<CL_PLATFORM_NAME>");

  return act_device;
}

void oclPrintDeviceInfo(const cl::Device &device, std::ostream &s, bool short_info)
{
	s << std::endl << "OpenCL Device info:" << std::endl
		<< "  CL_DEVICE_TYPE: " << printDeviceType(device.getInfo<CL_DEVICE_TYPE>()) << std::endl
		<< "  CL_VENDOR_NAME: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl
		<< "  CL_DEVICE_NAME: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	if (short_info) return;
	s << "  CL_DEVICE_IMAGE_SUPPORT: " << (device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() ? 1 : 0) << std::endl
		<< "  CL_DEVICE_COMPILER_AVAILABLE: " << (device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>() ? 1 : 0) << std::endl
		<< "  CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() << std::endl
		<< "  CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>() << std::endl
		<< "  CL_DEVICE_GLOBAL_MEM_SIZE: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl
		<< "  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << std::endl
		<< "  CL_DEVICE_LOCAL_MEM_TYPE: " << ((device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() == CL_LOCAL) ? "local" : "global") << std::endl
		<< "  CL_DEVICE_LOCAL_MEM_SIZE: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl
		<< "  CL_DEVICE_MAX_CLOCK_FREQUENCY: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl
		<< "  CL_DEVICE_MAX_COMPUTE_UNITS: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl
		<< "  CL_DEVICE_MAX_CONSTANT_ARGS: " << device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>() << std::endl
		<< "  CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl
		<< "  CL_DEVICE_MAX_WORK_GROUP_SIZE: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl
		<< "  CL_DEVICE_IMAGE2D_MAX_SIZE: " << device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>() << "x" << device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() << std::endl
		<< "  CL_DEVICE_MAX_WORK_ITEM_SIZES: ";
  std::vector<size_t> max_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  for(int j = 0; j < max_sizes.size(); j++)
  {
    if(j != 0) s << "x";
    s << max_sizes[j];
  }
  s << std::endl;
  std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
  
  /*if (extensions.find("cl_amd_device_attribute_query") != std::string::npos)
  {
      s << "  device CL_DEVICE_WAVEFRONT_WIDTH_AMD: " << device.getInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>() << std::endl
          << "  device CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD: " << device.getInfo< CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD>() << std::endl
          << "  device CL_DEVICE_LOCAL_MEM_BANKS_AMD: " << device.getInfo<CL_DEVICE_LOCAL_MEM_BANKS_AMD>() << std::endl
          //<< "  device CL_DEVICE_GFXIP__AMD: " << device.getInfo<CL_DEVICE_GFXIP_MAJOR_AMD>() << "." << device.getInfo<CL_DEVICE_GFXIP_MINOR_AMD>() << std::endl
          //<< "  device CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD: " << device.getInfo<CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD>() << std::endl
          //<< "  device CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD>() << std::endl
          << "  device CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD: " << device.getInfo<CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD>() << std::endl
          << "  device CL_DEVICE_SIMD_WIDTH_AMD: " << device.getInfo<CL_DEVICE_SIMD_WIDTH_AMD>() << std::endl
          //<< "  device CL_DEVICE_BOARD_NAME_AMD: " << device.getInfo<CL_DEVICE_BOARD_NAME_AMD>() << std::endl
          //<< "  device CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD: " << device.getInfo<CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD>() << std::endl
          << "  device CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD>() << std::endl
          << "  device CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD>() << std::endl
          << "  device CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD>() << std::endl;
  }*/
  if (extensions.find("cl_nv_device_attribute_query") != std::string::npos)
  {
      s << "  CL_DEVICE_WARP_SIZE_NV: " << device.getInfo<CL_DEVICE_WARP_SIZE_NV>() << std::endl
          << "  CL_DEVICE_REGISTERS_PER_BLOCK_NV: " << device.getInfo<CL_DEVICE_REGISTERS_PER_BLOCK_NV>() << std::endl
          << "  CL_DEVICE_COMPUTE_CAPABILITY__NV: " << device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>() << "." << device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>() << std::endl
          << "  CL_DEVICE_GPU_OVERLAP_NV: " << device.getInfo<CL_DEVICE_GPU_OVERLAP_NV>() << std::endl
          << "  CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV: " << device.getInfo<CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV>() << std::endl
          << "  CL_DEVICE_INTEGRATED_MEMORY_NV: " << device.getInfo<CL_DEVICE_INTEGRATED_MEMORY_NV>() << std::endl;
  }
  s << " CL_DEVICE_EXTENSIONS: " << extensions << std::endl << std::endl;
}

bool oclIsDeviceValid(const cl::Device &device)
{
  int32_t err_msg;
  cl_device_type dev_type;
  return device.getInfo(CL_DEVICE_TYPE, &dev_type) != CL_INVALID_DEVICE;
}

