use std::os::raw::*;
pub use ash::vk::{
    AllocationCallbacks as VkAllocationCallbacks,
    ComponentSwizzle as VkComponentSwizzle,
    Device as VkDevice,
    DeviceCreateInfo as VkDeviceCreateInfo,
    Filter as VkFilter,
    Format as VkFormat,
    Image as VkImage,
    ImageCreateFlags as VkImageCreateFlags,
    ImageUsageFlags as VkImageUsageFlags,
    Instance as VkInstance,
    InstanceCreateInfo as VkInstanceCreateInfo,
    PhysicalDevice as VkPhysicalDevice,
    Result as VkResult,
    SamplerAddressMode as VkSamplerAddressMode,
    SamplerMipmapMode as VkSamplerMipmapMode,
};
pub type EGLenum = c_uint;
pub type EGLContext = *mut c_void;
