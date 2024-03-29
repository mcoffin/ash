use generator::{
    variant_ident,
    Vulkan,
};

#[cfg(feature = "openxr")]
use generator::openxr::OpenXR;

#[test]
fn vulkan_variant_names_correct() {
    assert_eq!(
        variant_ident::<Vulkan>("VkCullModeFlagBits", "VK_CULL_MODE_FRONT_AND_BACK").to_string(),
        "FRONT_AND_BACK"
    );
}

#[cfg(feature = "openxr")]
#[test]
fn xr_variant_names_correct() {
    assert_eq!(
        variant_ident::<OpenXR>("XrStructureType", "XR_TYPE_INSTANCE_CREATE_INFO").to_string(),
        "INSTANCE_CREATE_INFO"
    );
    assert_eq!(
        variant_ident::<OpenXR>("XrViewConfigurationType", "XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO").to_string(),
        "PRIMARY_STEREO"
    );
    assert_eq!(
        variant_ident::<OpenXR>("XrSwapchainUsageFlagBits", "XR_SWAPCHAIN_USAGE_SAMPLED_BIT").to_string(),
        "SAMPLED"
    );
}
