include!("../../ash/src/vk/platform_types.rs");
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
    PFN_vkGetInstanceProcAddr,
};
pub type EGLenum = c_uint;
pub type EGLContext = *mut c_void;
pub type EGLDisplay = *mut c_void;
pub type EGLConfig = *mut c_void;
