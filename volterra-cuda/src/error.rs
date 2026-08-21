//! Error type for `volterra-cuda`, mirroring `cartan-cuda`'s `CudaError`.

use core::fmt;

use cuda_core::DriverError;
use cuda_host::EmbeddedModuleError;

#[derive(Debug)]
pub enum CudaError {
    /// A driver call failed. Error 803 is the usual one: the loaded kernel
    /// module and the userspace driver are different versions.
    Driver(DriverError),
    /// The PTX compiled from this crate could not be loaded onto the device.
    Module(EmbeddedModuleError),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver call failed: {e}"),
            Self::Module(e) => write!(f, "loading the compiled module failed: {e}"),
        }
    }
}

impl std::error::Error for CudaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(e) => Some(e),
            Self::Module(e) => Some(e),
        }
    }
}

impl From<DriverError> for CudaError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<EmbeddedModuleError> for CudaError {
    fn from(e: EmbeddedModuleError) -> Self {
        Self::Module(e)
    }
}
