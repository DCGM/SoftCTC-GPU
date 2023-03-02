#include "CudaHelper.h"

//Vrati retezec pro opencl error kod
const char *get_cuda_error_string(cudaError msg_id)
{
    return(cudaGetErrorString(msg_id));
  /*switch (msg_id)
  {
  case cudaSuccess:
    return "cudaSuccess";
  case cudaErrorInvalidValue:
    return "cudaErrorInvalidValue";
  case cudaErrorMemoryAllocation:
    return "cudaErrorMemoryAllocation";
  case cudaErrorInitializationError:
    return "cudaErrorInitializationError";
  case cudaErrorCudartUnloading:
    return "cudaErrorCudartUnloading";
  case cudaErrorProfilerDisabled:
    return "cudaErrorProfilerDisabled";
  case cudaErrorProfilerNotInitialized:
    return "cudaErrorProfilerNotInitialized";
  case cudaErrorProfilerAlreadyStarted:
    return "cudaErrorProfilerAlreadyStarted";
  case cudaErrorProfilerAlreadyStopped:
    return "cudaErrorProfilerAlreadyStopped";
  case cudaErrorInvalidConfiguration:
    return "cudaErrorInvalidConfiguration";
  case cudaErrorInvalidPitchValue:
    return "cudaErrorInvalidPitchValue";
  case cudaErrorInvalidSymbol:
    return "cudaErrorInvalidSymbol";
  case cudaErrorInvalidHostPointer:
    return "cudaErrorInvalidHostPointer";
  case cudaErrorInvalidDevicePointer:
    return "cudaErrorInvalidDevicePointer";
  case cudaErrorInvalidTexture:
    return "cudaErrorInvalidTexture";
  case cudaErrorInvalidTextureBinding:
    return "cudaErrorInvalidTextureBinding";
  case cudaErrorInvalidChannelDescriptor:
    return "cudaErrorInvalidChannelDescriptor";
  case cudaErrorInvalidMemcpyDirection:
    return "cudaErrorInvalidMemcpyDirection";
  case cudaErrorAddressOfConstant:
    return "cudaErrorAddressOfConstant";
  case cudaErrorTextureFetchFailed:
    return "cudaErrorTextureFetchFailed";
  case cudaErrorTextureNotBound:
    return "cudaErrorTextureNotBound";
  case cudaErrorSynchronizationError:
    return "cudaErrorSynchronizationError";
  case cudaErrorInvalidFilterSetting:
    return "cudaErrorInvalidFilterSetting";
  case cudaErrorInvalidNormSetting:
    return "cudaErrorInvalidNormSetting";
  case cudaErrorMixedDeviceExecution:
    return "cudaErrorMixedDeviceExecution";
  case cudaErrorNotYetImplemented:
    return "cudaErrorNotYetImplemented";
  case cudaErrorMemoryValueTooLarge:
    return "cudaErrorMemoryValueTooLarge";
  case cudaErrorStubLibrary:
    return "cudaErrorStubLibrary";
  case cudaErrorInsufficientDriver:
    return "cudaErrorInsufficientDriver";
  case cudaErrorCallRequiresNewerDriver:
    return "cudaErrorCallRequiresNewerDriver";
  case cudaErrorInvalidSurface:
    return "cudaErrorInvalidSurface";
  case cudaErrorDuplicateVariableName:
    return "cudaErrorDuplicateVariableName";
  case cudaErrorDuplicateTextureName:
    return "cudaErrorDuplicateTextureName";
  case cudaErrorDuplicateSurfaceName:
    return "cudaErrorDuplicateSurfaceName";
  case cudaErrorDevicesUnavailable:
    return "cudaErrorDevicesUnavailable";
  case cudaErrorIncompatibleDriverContext:
    return "cudaErrorIncompatibleDriverContext";
  case cudaErrorMissingConfiguration:
    return "cudaErrorMissingConfiguration";
  case cudaErrorPriorLaunchFailure:
    return "cudaErrorPriorLaunchFailure";
  case cudaErrorLaunchMaxDepthExceeded:
    return "cudaErrorLaunchMaxDepthExceeded";
  case cudaErrorLaunchFileScopedTex:
    return "cudaErrorLaunchFileScopedTex";
  case cudaErrorLaunchFileScopedSurf:
    return "cudaErrorLaunchFileScopedSurf";
  case cudaErrorSyncDepthExceeded:
    return "cudaErrorSyncDepthExceeded";
  case cudaErrorLaunchPendingCountExceeded:
    return "cudaErrorLaunchPendingCountExceeded";
  case cudaErrorInvalidDeviceFunction:
    return "cudaErrorInvalidDeviceFunction";
  case cudaErrorNoDevice:
    return "cudaErrorNoDevice";
  case cudaErrorInvalidDevice:
    return "cudaErrorInvalidDevice";
  case cudaErrorDeviceNotLicensed:
      return "cudaErrorDeviceNotLicensed";
  case cudaErrorSoftwareValidityNotEstablished:
      return "cudaErrorSoftwareValidityNotEstablished";
  case cudaErrorStartupFailure:
      return "cudaErrorStartupFailure";
  case cudaErrorInvalidKernelImage:
      return "cudaErrorInvalidKernelImage";
  case cudaErrorDeviceUninitialized:
      return "cudaErrorDeviceUninitialized";
  case cudaErrorMapBufferObjectFailed:
      return "cudaErrorMapBufferObjectFailed";
  case cudaErrorUnmapBufferObjectFailed:
      return "cudaErrorUnmapBufferObjectFailed";
  case cudaErrorArrayIsMapped:
      return "cudaErrorArrayIsMapped";
  case cudaErrorAlreadyMapped:
      return "cudaErrorAlreadyMapped";
  case cudaErrorNoKernelImageForDevice:
      return "cudaErrorNoKernelImageForDevice";
  case cudaErrorAlreadyAcquired:
      return "cudaErrorAlreadyAcquired";
  case cudaErrorNotMapped:
      return "cudaErrorNotMapped";
  case cudaErrorNotMappedAsArray:
      return "cudaErrorNotMappedAsArray";
  case cudaErrorNotMappedAsPointer:
      return "cudaErrorNotMappedAsPointer";
  case cudaErrorECCUncorrectable:
      return "cudaErrorECCUncorrectable";
  case cudaErrorUnsupportedLimit:
      return "cudaErrorUnsupportedLimit";
  case cudaErrorDeviceAlreadyInUse:
      return "cudaErrorDeviceAlreadyInUse";
  case cudaErrorPeerAccessUnsupported:
      return "cudaErrorPeerAccessUnsupported";
  case cudaErrorInvalidPtx:
      return "cudaErrorInvalidPtx";
  case cudaErrorInvalidGraphicsContext:
      return "cudaErrorInvalidGraphicsContext";
  case cudaErrorNvlinkUncorrectable:
      return "cudaErrorNvlinkUncorrectable";
  case cudaErrorJitCompilerNotFound:
      return "cudaErrorJitCompilerNotFound";
  case cudaErrorUnsupportedPtxVersion:
      return "cudaErrorUnsupportedPtxVersion";
  case cudaErrorJitCompilationDisabled:
      return "cudaErrorJitCompilationDisabled";
  case cudaErrorInvalidSource:
      return "cudaErrorInvalidSource";
  case cudaErrorFileNotFound:
      return "cudaErrorFileNotFound";
  case cudaErrorSharedObjectSymbolNotFound:
      return "cudaErrorSharedObjectSymbolNotFound";
  case cudaErrorSharedObjectInitFailed:
      return "cudaErrorSharedObjectInitFailed";
  case cudaErrorOperatingSystem:
      return "cudaErrorOperatingSystem";
  case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";
  case cudaErrorIllegalState:
      return "cudaErrorIllegalState";
  case cudaErrorSymbolNotFound:
      return "cudaErrorSymbolNotFound";
  case cudaErrorNotReady:
      return "cudaErrorNotReady";
  case cudaErrorIllegalAddress:
      return "cudaErrorIllegalAddress";
  case cudaErrorLaunchOutOfResources:
      return "cudaErrorLaunchOutOfResources";
  case cudaErrorLaunchTimeout:
      return "cudaErrorLaunchTimeout";
  case cudaErrorLaunchIncompatibleTexturing:
      return "cudaErrorLaunchIncompatibleTexturing";
  case cudaErrorPeerAccessAlreadyEnabled:
      return "cudaErrorPeerAccessAlreadyEnabled";
  case cudaErrorPeerAccessNotEnabled:
      return "cudaErrorPeerAccessNotEnabled";
  case cudaErrorSetOnActiveProcess:
      return "cudaErrorSetOnActiveProcess";
  case cudaErrorContextIsDestroyed:
      return "cudaErrorContextIsDestroyed";
  case cudaErrorAssert:
      return "cudaErrorAssert";
  case cudaErrorTooManyPeers:
      return "cudaErrorTooManyPeers";
  case cudaErrorHostMemoryAlreadyRegistered:
      return "cudaErrorHostMemoryAlreadyRegistered";
  case cudaErrorHostMemoryNotRegistered:
      return "cudaErrorHostMemoryNotRegistered";
  case cudaErrorHardwareStackError:
      return "cudaErrorHardwareStackError";
  case cudaErrorIllegalInstruction:
      return "cudaErrorIllegalInstruction";
  case cudaErrorMisalignedAddress:
      return "cudaErrorMisalignedAddress";
  case cudaErrorInvalidAddressSpace:
      return "cudaErrorInvalidAddressSpace";
  case cudaErrorInvalidPc:
      return "cudaErrorInvalidPc";
  case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";
  case cudaErrorCooperativeLaunchTooLarge:
      return "cudaErrorCooperativeLaunchTooLarge";
  case cudaErrorNotPermitted:
      return "cudaErrorNotPermitted";
  case cudaErrorNotSupported:
      return "cudaErrorNotSupported";
  case cudaErrorSystemNotReady:
      return "cudaErrorSystemNotReady";
  case cudaErrorSystemDriverMismatch:
      return "cudaErrorSystemDriverMismatch";
  case cudaErrorCompatNotSupportedOnDevice:
      return "cudaErrorCompatNotSupportedOnDevice";
  case cudaErrorStreamCaptureUnsupported:
      return "cudaErrorStreamCaptureUnsupported";
  case cudaErrorStreamCaptureInvalidated:
      return "cudaErrorStreamCaptureInvalidated";
  case cudaErrorStreamCaptureMerge:
      return "cudaErrorStreamCaptureMerge";
  case cudaErrorStreamCaptureUnmatched:
      return "cudaErrorStreamCaptureUnmatched";
  case cudaErrorStreamCaptureUnjoined:
      return "cudaErrorStreamCaptureUnjoined";
  case cudaErrorStreamCaptureIsolation:
      return "cudaErrorStreamCaptureIsolation";
  case cudaErrorStreamCaptureImplicit:
      return "cudaErrorStreamCaptureImplicit";
  case cudaErrorCapturedEvent:
      return "cudaErrorCapturedEvent";
  case cudaErrorStreamCaptureWrongThread:
      return "cudaErrorStreamCaptureWrongThread";
  case cudaErrorTimeout:
      return "cudaErrorTimeout";
  case cudaErrorGraphExecUpdateFailure:
      return "cudaErrorGraphExecUpdateFailure";
  case cudaErrorUnknown:
      return "cudaErrorUnknown";
  case cudaErrorApiFailureBase:
      return "cudaErrorApiFailureBase";
  default:
    return "Unknown";
  }*/
}

bool cudaPrintErrorExit(cudaError err_num, const char *text)
{
  if(err_num != cudaSuccess)
  {
    std::cerr << "Error: " << text << ": (" << err_num << ") " << get_cuda_error_string(err_num) << std::endl;
    std::string input_data;
    std::cin >> input_data;
    exit(1);
    return false;
  }
  return true;
}

bool cudaPrintError(cudaError err_num, const char *text, std::ostream *error_stream)
{
  if((err_num != cudaSuccess) && (error_stream != NULL)) *error_stream << "Error: " << text << ": (" << err_num << ") " << get_cuda_error_string(err_num) << std::endl;
  return (err_num == cudaSuccess);
}

bool cudaPrintInfo(cudaError err_num, const char *text, std::ostream *error_stream)
{
  if((err_num != cudaSuccess) && (error_stream != NULL)) *error_stream << "Info: " << text << ": (" << err_num << ") " << get_cuda_error_string(err_num) << std::endl;
  return (err_num == cudaSuccess);
}

double getCudaEventTime(cudaEvent_t start, cudaEvent_t stop)
{
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms/1000.0;
}