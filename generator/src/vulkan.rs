use crate::config::*;
use quote::*;
use proc_macro2::{
    Ident,
    TokenStream,
};

#[derive(Debug, Clone, Copy)]
pub struct Vulkan;

impl Vulkan {
    const FUNCTIONS: FunctionsConfig<'static> = FunctionsConfig {
        static_fns: &[
            "vkGetInstanceProcAddr",
        ],
        entry_fns: &[
            "vkCreateInstance",
            "vkEnumerateInstanceLayerProperties",
            "vkEnumerateInstanceExtensionProperties",
            "vkEnumerateInstanceVersion",
        ],
        dispatch_fns: &[
            ("vkGetDeviceProcAddr", "Instance"),
        ],
        dispatch_types: &[
            "Instance",
            "Device",
        ],
        dispatch_hints: &[
            ("CommandBuffer", "Device"),
            ("Queue", "Device"),
        ],
    };
}

impl ApiConfig for Vulkan {
    const NAME: &'static str = "vulkan";
    const MODULE_NAME: &'static str = "vk";
    const REGISTRY_FILENAME: &'static str = "vk.xml";
    const TYPE_PREFIX: &'static str = "Vk";
    const COMMAND_PREFIX: &'static str = "vk";
    const CONSTANT_PREFIX: &'static str = "VK_";
    const TAGGED_STRUCT: StructConfig<'static> = StructConfig {
        ty: "VkStructureType",
        ty_name: "s_type",
        next_name: "p_next",
    };
    const RESULT_ENUM: ResultConfig<'static> = ResultConfig {
        ty: "VkResult",
        prefix: None,
    };
    const VENDOR_SUFFIXES: &'static [&'static str] = &[
        "AMD",
        "AMDX",
        "ANDROID",
        "ARM",
        "BRCM",
        "CHROMIUM",
        "EXT",
        "FB",
        "FSL",
        "FUCHSIA",
        "GGP",
        "GOOGLE",
        "HUAWEI",
        "IMG",
        "INTEL",
        "JUICE",
        "KDAB",
        "KHR",
        "KHX",
        "LUNARG",
        "MESA",
        "MSFT",
        "MVK",
        "NN",
        "NV",
        "NVX",
        "NXP",
        "NZXT",
        "QCOM",
        "QNX",
        "RASTERGRID",
        "RENDERDOC",
        "SAMSUNG",
        "SEC",
        "TIZEN",
        "VALVE",
        "VIV",
        "VSI",
    ];
    fn function_type<'a>(f: &'a vk_parse::CommandDefinition) -> FunctionType<'a> {
        const DEFAULT_TYPE: FunctionType<'static> = FunctionType::Dispatched("Instance");
        Self::FUNCTIONS.function_type(f, |s| s.strip_prefix(Self::TYPE_PREFIX).unwrap_or(s))
            .unwrap_or_else(|e| {
                warn!("{}. defaulting to {:?}", &e, &DEFAULT_TYPE);
                DEFAULT_TYPE
            })
    }
    #[inline(always)]
    fn function_pointer_name(fn_name: &str) -> Ident {
        format_ident!("PFN_{}", fn_name)
    }
    fn manual_struct(struct_name: &str) -> Option<TokenStream> {
        match struct_name {
            "VkTransformMatrixKHR" => {
                Some(quote! {
                    #[repr(C)]
                    #[derive(Copy, Clone)]
                    pub struct TransformMatrixKHR {
                        pub matrix: [f32; 12],
                    }
                })
            },
            "VkAccelerationStructureInstanceKHR" => {
                Some(quote! {
                    #[repr(C)]
                    #[derive(Copy, Clone)]
                    pub union AccelerationStructureReferenceKHR {
                        pub device_handle: DeviceAddress,
                        pub host_handle: AccelerationStructureKHR,
                    }
                    #[repr(C)]
                    #[derive(Copy, Clone)]
                    #[doc = "<https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureInstanceKHR.html>"]
                    pub struct AccelerationStructureInstanceKHR {
                        pub transform: TransformMatrixKHR,
                        /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
                        pub instance_custom_index_and_mask: Packed24_8,
                        /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
                        pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
                        pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
                    }
                })
            },
            "VkAccelerationStructureSRTMotionInstanceNV" => {
                Some(quote! {
                    #[repr(C)]
                    #[derive(Copy, Clone)]
                    #[doc = "<https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureSRTMotionInstanceNV.html>"]
                    pub struct AccelerationStructureSRTMotionInstanceNV {
                        pub transform_t0: SRTDataNV,
                        pub transform_t1: SRTDataNV,
                        /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
                        pub instance_custom_index_and_mask: Packed24_8,
                        /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
                        pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
                        pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
                    }
                })
            },
            "VkAccelerationStructureMatrixMotionInstanceNV" => {
                Some(quote! {
                    #[repr(C)]
                    #[derive(Copy, Clone)]
                    #[doc = "<https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/AccelerationStructureMatrixMotionInstanceNV.html>"]
                    pub struct AccelerationStructureMatrixMotionInstanceNV {
                        pub transform_t0: TransformMatrixKHR,
                        pub transform_t1: TransformMatrixKHR,
                        /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
                        pub instance_custom_index_and_mask: Packed24_8,
                        /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
                        pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
                        pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
                    }
                })
            },
            _ => None,
        }
    }
}
