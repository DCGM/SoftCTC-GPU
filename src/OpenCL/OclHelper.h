#ifndef OCL_HELPER_H
#define OCL_HELPER_H

#include <CL/cl.hpp>
#ifdef COMPUTE_CACHE_ENABLED
#ifdef BOOST_ENABLE
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#else
#include <filesystem>
#endif
#endif

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <climits>
#include <algorithm>
#include <fstream>

#ifdef max
  #undef max
#endif

#ifdef min
  #undef min
#endif

#ifndef CL_DEVICE_WAVEFRONT_WIDTH_AMD
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD 0x4043
#endif

const std::string compute_cache_dir("cache");

enum e_ocl_compile
{
  OCL_COMPILE_FAILED = 0,
  OCL_COMPILE_CACHED,
  OCL_COMPILE_COMPILED
};

bool read_file(const std::string &filename, std::string &out_data);

e_ocl_compile compileOpenclSource(cl_program &program, cl_context context, cl_device_id device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream = &std::cerr);
e_ocl_compile compileOpenclSource(cl::Program &program, const cl::Context &context, const cl::Device &device, const std::string &program_defines, const std::string &kernel_file_name, std::ostream *error_stream = &std::cerr);
e_ocl_compile compileOpenclContent(cl::Program& program, const cl::Context& context, const cl::Device& device, const std::string& program_defines, const std::string& program_content, std::ostream* error_stream = &std::cerr);

enum e_device_type
{
  DEVICE_TYPE_OTHER = 0,
  DEVICE_TYPE_NVIDIA_GPU,
  DEVICE_TYPE_NVIDIA_CPU,
  DEVICE_TYPE_AMD_GPU,
  DEVICE_TYPE_AMD_CPU,
  DEVICE_TYPE_INTEL_CPU,
  DEVICE_TYPE_INTEL_GPU
};

// implementation of opencl types operators
bool operator== (const cl_uchar3 &c1, const cl_uchar3 &c2);
bool operator== (const cl_float3 &c1, const cl_float3 &c2);
bool operator== (const cl_double3 &c1, const cl_double3 &c2);
bool operator!= (const cl_uchar3 &c1, const cl_uchar3 &c2);
bool operator!= (const cl_float3 &c1, const cl_float3 &c2);
bool operator!= (const cl_double3 &c1, const cl_double3 &c2);

extern std::string empty_string;

cl::Device oclSelectDeviceByStdin(const std::vector<cl::Device> &devices); // set device id from input
void oclPrintDevices(const std::vector<cl::Device> &devices); // print opencl devices
cl::Device oclGetDeviceFromParameters(cl_device_type device_type = CL_DEVICE_TYPE_ALL, const std::string &device_vendor = empty_string, const std::string &device_name = empty_string); // get specified opencl device
std::vector<cl::Device> oclGetDevices(cl_device_type device_type = CL_DEVICE_TYPE_ALL); // get available opencl devices
std::string oclGetDeviceSpecificDefines(const cl::Device &device); // get device specific defines
void oclPrintDeviceInfo(const cl::Device &device, std::ostream &s, bool short_info = false);
bool oclIsDeviceValid(const cl::Device &device);

size_t getMaximumGpuThreads(const cl::Device &device);

bool oclPrintErrorExit(int32_t msg_id, const char *msg); // check error
bool oclPrintError(int32_t err_num, const char *text, std::ostream *error_stream = &std::cerr);
bool oclPrintInfo(int32_t err_num, const char *text, std::ostream *error_stream = &std::cerr);

double getEventTime(cl_event i_event); // get gpu time
double getEventTime(cl::Event &event); // get gpu time

std::string read_file(const std::string &filename); // load file to string
bool read_file(const std::string &filename, std::string &out_data); // load file to string

// argument setter
template <typename T0>
void oclSetKernelArgs(cl::Kernel &kernel, int32_t act_pos, T0 act_arg)
{
  kernel.setArg(act_pos, act_arg);
}

// argument setter
template <typename T0, typename ...TN>
void oclSetKernelArgs(cl::Kernel &kernel, int32_t act_pos, T0 act_arg, TN... rest_args)
{
  kernel.setArg(act_pos, act_arg);
  oclSetKernelArgs(kernel, act_pos + 1, rest_args...);
}

template <typename T>
T ceilDivBy(T id, T div)
{
  return (id + div - 1) / div;
}

template <typename T>
T alignTo(T id, T align)
{
  return ceilDivBy(id, align) * align;
}

class KernelRange
{
public:
  uint32_t x;
  uint32_t y;
  uint32_t z;

  KernelRange(uint32_t x = 0, uint32_t y = 0, uint32_t z = 0);
  KernelRange(uint32_t x, uint32_t y, uint32_t z, KernelRange &local);

  bool isAligned(KernelRange &local);

  void ceilRangeDivBy(KernelRange &local);
  bool isValid();
  void alignRangeTo(KernelRange &local);

  cl::NDRange toNDRange();

  KernelRange operator +(const KernelRange &data);
  KernelRange operator -(const KernelRange &data);
  bool operator ==(const KernelRange &data);

  uint32_t getFlatSize();
  bool isZero();
  KernelRange getOverlappingSizeArea(const KernelRange &limit);
  void zeroOverlappingArea(const KernelRange &limit);
};

class KernelRangeData
{
public:
  KernelRange local;
  KernelRange global;
  KernelRange offset;

  KernelRangeData(KernelRange &total_size, KernelRange &local_size, uint32_t max_size);

  bool isBegin();
  void init();
  void next();

private:
  bool isValid();
  KernelRange getMaxKernelRange(uint32_t max_size);

  KernelRange act_start;
  KernelRange total_size;
  KernelRange max_kernel_size;
};

class OclKernel: public cl::Kernel
{
public:
  OclKernel(const cl::Program& program, const char* name, const cl::Device& device, int const_shared = 0, int shared_per_thread = 0, int min_shared_per_group = 0, int force_align_to_size = 1);

  KernelRange getGroupSize();
  uint32_t getDynamicMemSize();

  template <typename ...TN>
  inline void setArgs(TN... args)
  {
    oclSetKernelArgs(*(cl::Kernel *)this, 0, args...);
    args_count = sizeof...(TN);
  };

  template <typename ...TN>
  inline void pushBackArgs(TN... args)
  {
    oclSetKernelArgs(*(cl::Kernel *)this, args_count, args...);
    args_count += sizeof...(TN);
  };

  inline void clearArgs()
  {
    args_count = 0;
  }

private:
  int32_t err_msg;
  uint32_t max_group_size;
  uint32_t dynamic_mem_size;
  uint32_t args_count;
  KernelRange group_size;

  void setMaxGroupSize(const cl::Device& device, int const_shared, int shared_per_thread, int min_shared_per_group, int force_align_to_size);
};

template <typename T>
class ImageData
{
public:
	uint32_t width = 0;
	uint32_t height = 0;
	T *ptr = NULL;
	cl_int getLinearSize() const
	{
		return this->width * this->height;
	}
	cl_int2 getSize() const
	{
		return { (int32_t)this->width, (int32_t)this->height };
	}
	size_t getBufferSize() const
	{
		return this->width * this->height * sizeof(T);
	}
	int32_t getAllocatedSize() const
	{
		return this->allocated_pixel_count;
	}
	size_t getAllocatedBufferSize() const
	{
		return this->allocated_pixel_count * sizeof(T);
	}
protected:
	ImageData(){}
	virtual ~ImageData(){};
	size_t allocated_pixel_count = 0;
};

template <typename T>
class CpuImageData : public ImageData<T>
{
public:
  CpuImageData(){}
  virtual ~CpuImageData(){}
  virtual void resize(uint32_t width = 0, uint32_t height = 0) = 0;
};

template <typename T>
class CpuImageVector : public CpuImageData<T>
{
public:
	std::vector<T> data;
	CpuImageVector(uint32_t width = 0, uint32_t height = 0)
	{
		this->resize(width, height);
	}
	CpuImageVector(cl_int2 size)
	{
		this->resize(size.x, size.y);
	}
	~CpuImageVector()
	{
		this->data.clear();
	}
	void resize(uint32_t width, uint32_t height)
	{
		this->width = width;
		this->height = height;
		this->data.resize(width * height);
		this->ptr = data.data();
		this->allocated_pixel_count = data.capacity();
	}
	void fill(T *data)
	{
		for (int i = 0; i < this->width * this->height; i++)
		{
			this->data[i] = data[i];
		}
	}
};


template <typename T>
class CpuImagePtr : public CpuImageData<T>
{
public:
	CpuImagePtr(T *data = NULL, uint32_t width = 0, uint32_t height = 0)
	{
		this->ptr = data;
		this->allocated_pixel_count = width * height;
		this->resize(width, height);
	}
	CpuImagePtr(cl_int2 size, T *data) : CpuImageData<T>()
	{
		this->ptr = data;
		this->allocated_pixel_count = size.x * size.y;
		this->resize(size.x, size.y);
	}
	~CpuImagePtr()
	{
	}
	void fill(T *data)
	{
		for (int i = 0; i < this->width * this->height; i++)
		{
			this->data[i] = data[i];
		}
	}
	void resize(uint32_t width, uint32_t height)
	{
		if (width * height < this->allocated_pixel_count)
		{
			this->width = width;
			this->height = height;
		}
	}
};


enum e_buf_type
{
  BUF_TYPE_TEXTURE2D,
  BUF_TYPE_LINEAR
};

inline std::string toString(e_buf_type buf_type)
{
  switch(buf_type)
  {
  case BUF_TYPE_LINEAR:
    return std::string("global");
    break;
  case BUF_TYPE_TEXTURE2D:
    return std::string("texture");
    break;
  default:
    return std::string();
  }
}

template <typename T>
class GpuImageData : public ImageData<T>
{
public:
  typedef T TemplateT;
  e_buf_type type;
  cl::Memory *data = NULL;
  
  GpuImageData(e_buf_type in_type) : type(in_type){}

  virtual ~GpuImageData()
  {
    deleteBuffer();
  }
  
  virtual bool copyFrom(const CpuImageData<T> &cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) = 0;
  virtual bool copyFrom(const GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) = 0;
  virtual bool copyMappedFrom(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) = 0;
  virtual bool copyMappedFrom(T *gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) = 0;
  virtual bool copyFrom(const T *cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) = 0;
  virtual bool copyTo(CpuImageData<T> &cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) const = 0;
  virtual bool copyTo(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) const = 0;
  virtual bool copyTo(T *cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL, uint32_t width = INT_MAX, uint32_t height = INT_MAX) const = 0;
  virtual bool copyMappedTo(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) = 0;
  virtual bool fill(T &fill_data, const cl::CommandQueue &queue, cl::Event *event = NULL) = 0;
  virtual bool map(const cl::CommandQueue &queue, cl_map_flags flags) = 0;
  virtual bool unmap(const cl::CommandQueue &queue) = 0;

  virtual void resize(uint32_t width, uint32_t height, const cl::CommandQueue &queue, cl::Event *event = NULL) = 0;

  bool isMapped()
  {
	  return this->ptr != NULL;
  }
protected:
  void deleteBuffer()
  {
#ifdef DEBUG_GPU_MEM
	if(data != NULL) std::cerr << "Delete GpuImageData of size " << this->allocated_pixel_count << std::endl;
#endif
    delete data;
    this->data = NULL;
	this->ptr = NULL;
  }
  
  cl_map_flags map_flags = CL_MAP_READ | CL_MAP_WRITE;
};

template <typename T>
class GpuImageTexture2D: public GpuImageData<T>
{
public:
  GpuImageTexture2D(uint32_t width, uint32_t height, const cl::Context &context, cl_mem_flags mem_flags, cl_channel_order in_channel_order, cl_channel_type in_channel_type): GpuImageData<T>(BUF_TYPE_TEXTURE2D), channel_order(in_channel_order), channel_type(in_channel_type)
  {
    createBuffer(width, height, context, mem_flags);
  }
  GpuImageTexture2D(uint32_t width, const cl::Context &context, cl_mem_flags mem_flags, cl_channel_order in_channel_order, cl_channel_type in_channel_type) : GpuImageData<T>(BUF_TYPE_TEXTURE2D), channel_order(in_channel_order), channel_type(in_channel_type)
  {
	createBuffer(width, 1u, context, mem_flags);
  }
  bool copyFrom(const CpuImageData<T> &cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
	  if ((cpu_data.width != this->width) || (cpu_data.height != this->height))
	  {
#ifdef DEBUG_GPU_MEM
		  std::cerr << "GpuImageBuffer->copyFrom CpuImageData failed " << this->width << "x" << this->height << "!=" << cpu_data.width << "x" << cpu_data.height << std::endl;
#endif
		  return false;
	  }
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = this->width;
    region[1] = this->height;
    region[2] = 1;
    oclPrintErrorExit(queue.enqueueWriteImage(*(cl::Image2D *)this->data, CL_FALSE, origin, region, 0, 0, cpu_data.ptr, NULL, event), "GpuImageData: cpu_data.data -> data");
    return true;
  }
  bool copyFrom(const T *cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = this->width;
    region[1] = this->height;
    region[2] = 1;
    oclPrintErrorExit(queue.enqueueWriteImage(*(cl::Image2D *)this->data, CL_FALSE, origin, region, 0, 0, (void *)cpu_data, NULL, event), "GpuImageData: cpu_data -> data");
    return true;
  }
  bool copyTo(CpuImageData<T> &cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) const
  {
	  if ((cpu_data.width != this->width) || (cpu_data.height != this->height))
	  {
#ifdef DEBUG_GPU_MEM
		  std::cerr << "GpuImageBuffer->copyTo CpuImageData failed " << this->width << "x" << this->height << "!=" << cpu_data.width << "x" << cpu_data.height << std::endl;
#endif
		  return false;
	  }
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = this->width;
    region[1] = this->height;
    region[2] = 1;
    oclPrintErrorExit(queue.enqueueReadImage(*(cl::Image2D *)this->data, CL_FALSE, origin, region, 0, 0, cpu_data.ptr, NULL, event), "GpuImageData: data -> cpu_data.data");
    return true;
  }
  bool copyTo(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) const
  {
	  if ((gpu_data.width != this->width) || (gpu_data.height != this->height))
	  {
#ifdef DEBUG_GPU_MEM
		  std::cerr << "GpuImageBuffer->copyTo GpuImageData failed " << this->width << "x" << this->height << "!=" << gpu_data.width << "x" << gpu_data.height << std::endl;
#endif
		  return false;
	  }
    cl::size_t<3> origin;
    cl::size_t<3> region;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = this->width;
    region[1] = this->height;
    region[2] = 1;
    switch(gpu_data.type)
    {
    case BUF_TYPE_TEXTURE2D:
      oclPrintErrorExit(queue.enqueueCopyImage(*(cl::Image2D *)this->data, *(cl::Image2D *)gpu_data.data, origin, origin, region, NULL, event), "GpuImageData: data -> gpu_data.data");
      break;
    case BUF_TYPE_LINEAR:
      oclPrintErrorExit(queue.enqueueCopyImageToBuffer(*(cl::Image2D *)this->data, *(cl::Buffer *)gpu_data.data, origin, region, 0, NULL, event), "GpuImageData: data -> gpu_data.data");
      break;
    }
    return true;
  }
  bool copyFrom(const GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    return gpu_data.copyTo(*this, queue, event);
  }
  bool fill(T &pattern, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    //oclPrintErrorExit(queue.enqueueFillImage(*(cl::Image2D *)this->data, conv(pattern), 0, sizeof(T), width * height * sizeof(T)), "GpuImageData: fill data");
    return false;
  }
  bool copyTo(T *cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL, uint32_t width = INT_MAX, uint32_t height = INT_MAX) const
  {
    if(width * height == 0) return true;
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = std::min(width, this->width);
    region[1] = std::min(height, this->height);
    region[2] = 1;
    oclPrintErrorExit(queue.enqueueReadImage(*(cl::Image2D *)this->data, CL_FALSE, origin, region, 0, 0, cpu_data, NULL, event), "GpuImageData: data -> cpu_data.data");
    return true;
  }
  void resize(uint32_t width, uint32_t height, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    if((width > allocated_pixel_width) || (height > allocated_pixel_height))
    {
		cl_mem_flags mem_flags;
		cl::Context context;
		oclPrintErrorExit(this->data->getInfo(CL_MEM_CONTEXT, &context), "GpuImageData: data.getInfo -> CL_MEM_CONTEXT");
		oclPrintErrorExit(this->data->getInfo(CL_MEM_FLAGS, &mem_flags), "GpuImageData: data.getInfo -> CL_MEM_FLAGS");
		this->createBuffer(width, height, context, mem_flags);
    }
	else
	{
		this->width = width;
		this->height = height;
	}
  }
  bool map(const cl::CommandQueue &queue, cl_map_flags flags)
  {
	  return false;
  }

  bool unmap(const cl::CommandQueue &queue)
  {
	  return false;
  }

  bool copyMappedTo(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
	  return false;
  }

  bool copyMappedFrom(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
	  return false;
  }

  bool copyMappedFrom(T *gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
	  return false;
  }
private:
  void createBuffer(uint32_t width, uint32_t height, const cl::Context &context, cl_mem_flags mem_flags)
  {
	this->deleteBuffer();
	this->width = width;
	this->height = height;
	this->allocated_pixel_width = std::max(allocated_pixel_width, width);
	this->allocated_pixel_height = std::max(allocated_pixel_height, height);
	this->allocated_pixel_count = this->allocated_pixel_width * this->allocated_pixel_height;
    int32_t err_msg;
	if(this->allocated_pixel_count != 0)
	{
#ifdef DEBUG_GPU_MEM
		std::cerr << "Create GpuImageTexture2D of size " << allocated_pixel_width << "x" << allocated_pixel_height << std::endl;
#endif
		this->data = new cl::Image2D(context, mem_flags, cl::ImageFormat(channel_order, channel_type), allocated_pixel_width, allocated_pixel_height, 0, 0, &err_msg);
		oclPrintErrorExit(err_msg, "GpuImageData: cl::Buffer data");
	}
  }
  uint32_t allocated_pixel_width = 0;
  uint32_t allocated_pixel_height = 0;
  cl_channel_order channel_order;
  cl_channel_type channel_type;
};

template <typename T>
class GpuImageBuffer: public GpuImageData<T>
{
public:
  GpuImageBuffer(uint32_t width, uint32_t height, const cl::Context &context, cl_mem_flags mem_flags): GpuImageData<T>(BUF_TYPE_LINEAR)
  {
	createBuffer(width, height, context, mem_flags);
  }
  GpuImageBuffer(uint32_t width, const cl::Context &context, cl_mem_flags mem_flags) : GpuImageData<T>(BUF_TYPE_LINEAR)
  {
	createBuffer(width, 1, context, mem_flags);
  }
  ~GpuImageBuffer(){}
  bool copyFrom(const CpuImageData<T> &cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    if((cpu_data.width != this->width) || (cpu_data.height != this->height)) this->resize(cpu_data.width, cpu_data.height, queue);
    oclPrintErrorExit(queue.enqueueWriteBuffer(*(cl::Buffer *)this->data, CL_FALSE, 0, cpu_data.getBufferSize(), cpu_data.ptr, NULL, event), "GpuImageData: cpu_data.data -> data");
    return true;
  }
  bool copyFrom(const T *cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    oclPrintErrorExit(queue.enqueueWriteBuffer(*(cl::Buffer *)this->data, CL_FALSE, 0, this->getBufferSize(), cpu_data, NULL, event), "GpuImageData: cpu_data -> data");
    return true;
  }
  bool copyTo(CpuImageData<T> &cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) const
  {
	  if ((cpu_data.width != this->width) || (cpu_data.height != this->height))
	  {
#ifdef DEBUG_GPU_MEM
		  std::cerr << "GpuImageTexture2D->copyTo CpuImageData failed " << this->width << "x" << this->height << "!=" << cpu_data.width << "x" << cpu_data.height << std::endl;
#endif
		  return false;
	  }
    oclPrintErrorExit(queue.enqueueReadBuffer(*(cl::Buffer *)this->data, CL_FALSE, 0, cpu_data.getBufferSize(), cpu_data.ptr, NULL, event), "GpuImageData: data -> cpu_data.data");
    return true;
  }
  bool copyTo(T *cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL, uint32_t width = INT_MAX, uint32_t height = INT_MAX) const
  {
    if(width * height == 0) return true;
    width = std::min(width, this->width);
    height = std::min(height, this->height);
    if(width == this->width)
    {
      oclPrintErrorExit(queue.enqueueReadBuffer(*(cl::Buffer *)this->data, CL_FALSE, 0, this->getBufferSize(), cpu_data, NULL, event), "GpuImageData: data -> cpu_data");
    }
    else
    {
      for(int y = 0; y < height; y++)
      {
        oclPrintErrorExit(queue.enqueueReadBuffer(*(cl::Buffer *)this->data, CL_FALSE, y * this->width * sizeof(T), width * sizeof(T), cpu_data + y * width, NULL, event), "GpuImageData: data -> cpu_data");
      }
    }
    return true;
  }
  bool copyTo(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL) const
  {
	  if ((gpu_data.width != this->width) || (gpu_data.height != this->height))
	  {
#ifdef DEBUG_GPU_MEM
		  std::cerr << "GpuImageTexture2D->copyTo GpuImageData failed " << this->width << "x" << this->height << "!=" << gpu_data.width << "x" << gpu_data.height << std::endl;
#endif
		  return false;
	  }
    cl::size_t<3> origin;
    cl::size_t<3> region;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = this->width;
    region[1] = this->height;
    region[2] = 1;
    switch(gpu_data.type)
    {
    case BUF_TYPE_TEXTURE2D:
      oclPrintErrorExit(queue.enqueueCopyBufferToImage(*(cl::Buffer *)this->data, *(cl::Image2D *)gpu_data.data, 0, origin, region, NULL, event), "GpuImageData: data -> gpu_data.data");
      break;
    case BUF_TYPE_LINEAR:
      oclPrintErrorExit(queue.enqueueCopyBuffer(*(cl::Buffer *)this->data, *(cl::Buffer *)gpu_data.data, 0, 0, this->getBufferSize(), NULL, event), "GpuImageData: data -> gpu_data.data");
      break;
    }
    return true;
  }
  bool copyFrom(const GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    return gpu_data.copyTo(*this, queue, event);
  }
  bool fill(T &pattern, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    oclPrintErrorExit(queue.enqueueFillBuffer(*(cl::Buffer *)this->data, pattern, 0, this->getBufferSize(), NULL, event), "GpuImageData: fill data");
    return true;
  }
  void resize(uint32_t width, uint32_t height, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
    if(width * height > this->allocated_pixel_count)
    {
	  cl_mem_flags mem_flags;
	  cl::Context context;
      oclPrintErrorExit(this->data->getInfo(CL_MEM_CONTEXT, &context), "GpuImageData: data.getInfo -> CL_MEM_CONTEXT");
      oclPrintErrorExit(this->data->getInfo(CL_MEM_FLAGS, &mem_flags), "GpuImageData: data.getInfo -> CL_MEM_FLAGS");
	  bool mapped = this->isMapped();
	  this->unmap(queue);
      
      this->createBuffer(width, height, context, mem_flags);
	  if (mapped) this->map(queue, this->map_flags);
    }
	else
	{
		this->width = width;
		this->height = height;
	}
  }
  bool map(const cl::CommandQueue &queue, cl_map_flags flags)
  {
	  if((flags != this->map_flags) && (!this->unmap(queue))) return false;
	  if(this->ptr == NULL)
	  {
#ifdef DEBUG_GPU_MEM
		  std::cerr << "Map GpuImageBuffer of size " << this->allocated_pixel_count << ":" << this->width << "x" << this->height << std::endl;
#endif
		  int32_t err_msg;
		  this->ptr = (T *)queue.enqueueMapBuffer(*(cl::Buffer *)this->data, CL_FALSE, flags, 0, this->getAllocatedSize(), NULL, NULL, &err_msg);
		  this->map_flags = flags;
		  oclPrintErrorExit(err_msg, "GpuImageData: queue.enqueueMapBuffer");
	  }

	  return true;
  }

  bool unmap(const cl::CommandQueue &queue)
  {
	  if (this->ptr != NULL)
	  {
#ifdef DEBUG_GPU_MEM
		  std::cerr << "Unmap GpuImageBuffer of size " << this->allocated_pixel_count << ":" << this->width << "x" << this->height << std::endl;
#endif
		  oclPrintErrorExit(queue.enqueueUnmapMemObject(*this->data, this->ptr, NULL, NULL), "GpuImageData: queue.enqueueUnmapBuffer");
		  queue.finish();
		  this->ptr = NULL;
	  }
	  return true;
  }

  bool copyMappedTo(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
	  if ((gpu_data.width != this->width) || (gpu_data.height != this->height) || (!this->map(queue, CL_MAP_WRITE))) return false;
	  switch (gpu_data.type)
	  {
	  case BUF_TYPE_LINEAR:
		  oclPrintErrorExit(queue.enqueueWriteBuffer(*(cl::Buffer *)gpu_data.data, CL_FALSE, 0, this->getBufferSize(), (void *)this->ptr, NULL, event), "GpuImageData: queue.copyMappedTo");
		  break;
	  case BUF_TYPE_TEXTURE2D:
		  return false;
		  break;
	  }
	  return true;
  }

  bool copyMappedFrom(GpuImageData<T> &gpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
	  if ((gpu_data.width != this->width) || (gpu_data.height != this->height) || (!this->map(queue, CL_MAP_READ))) return false;
	  switch (gpu_data.type)
	  {
	  case BUF_TYPE_LINEAR:
		  oclPrintErrorExit(queue.enqueueReadBuffer(*(cl::Buffer *)gpu_data.data, CL_FALSE, 0, this->getBufferSize(), (void *)this->ptr, NULL, event), "GpuImageData: queue.copyMappedTo");
		  break;
	  case BUF_TYPE_TEXTURE2D:
		  return false;
		  break;
	  }
	  return true;
  }

  bool copyMappedFrom(T *cpu_data, const cl::CommandQueue &queue, cl::Event *event = NULL)
  {
	  if(!this->map(queue, CL_MAP_READ)) return false;
	  memcpy(this->ptr, cpu_data, this->width * this->height * sizeof(T));
	  return true;
  }

private:
  void createBuffer(uint32_t width, uint32_t height, const cl::Context &context, cl_mem_flags mem_flags)
  {
	this->deleteBuffer();
	uint32_t pixel_size = width * height;
	this->allocated_pixel_count = pixel_size;
	this->width = width;
	this->height = height;
    int32_t err_msg;
	if(this->allocated_pixel_count > 0)
	{
#ifdef DEBUG_GPU_MEM
		std::cerr << "Create GpuImageBuffer of size " << pixel_size << std::endl;
#endif
		this->data = new cl::Buffer(context, mem_flags, pixel_size * sizeof(T), 0, &err_msg);
		oclPrintErrorExit(err_msg, "GpuImageData: cl::Buffer data");
	}
  }
};

template <typename T>
inline GpuImageData<T> *getGpuImageData(e_buf_type buf_type, uint32_t width, uint32_t height, const cl::Context &context, cl_mem_flags mem_flags, cl_channel_order channel_order = CL_RGBA, cl_channel_type channel_type = CL_FLOAT)
{
  GpuImageData<T> *out_image = NULL;
  if(width == 0) return NULL;
  switch(buf_type)
  {
  case BUF_TYPE_LINEAR:
    out_image = new GpuImageBuffer<T>(width, height, context, mem_flags);
    break;
  case BUF_TYPE_TEXTURE2D:
    out_image = new GpuImageTexture2D<T>(width, height, context, mem_flags, channel_order, channel_type);
    break;
  }
  return out_image;
}

template <typename T>
inline GpuImageData<T> *getGpuImageData(e_buf_type buf_type, uint32_t width, const cl::Context &context, cl_mem_flags mem_flags, cl_channel_order channel_order = CL_RGBA, cl_channel_type channel_type = CL_FLOAT)
{
	GpuImageData<T> *out_image = NULL;
	if (width == 0) return NULL;
	switch (buf_type)
	{
	case BUF_TYPE_LINEAR:
		out_image = new GpuImageBuffer<T>(width, context, mem_flags);
		break;
	case BUF_TYPE_TEXTURE2D:
		out_image = new GpuImageTexture2D<T>(width, context, mem_flags, channel_order, channel_type);
		break;
	}
	return out_image;
}
#endif
