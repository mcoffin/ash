use crate::config::*;
use quote::*;
use proc_macro2::{
    Ident,
    TokenStream,
};
use std::path::Path;

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
    const ACCESSOR_BLACKLIST: [&'static str; 4] = [
        "VkBaseInStructure",
        "VkBaseOutStructure",
        "VkTransformMatrixKHR",
        "VkAccelerationStructureInstanceKHR",
    ];
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
        ty_name: "sType",
        next_name: "pNext",
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

    fn manual_derives(struct_name: &str) -> Option<TokenStream> {
        match struct_name {
            "VkClearRect" | "VkExtent2D" | "VkExtent3D" | "VkOffset2D" | "VkOffset3D" | "VkRect2D"
            | "VkSurfaceFormatKHR" => Some(quote! {PartialEq, Eq, Hash}),
            _ => None,
        }
    }

    #[inline(always)]
    fn generate_accessors(struct_name: &str) -> bool {
        !Self::ACCESSOR_BLACKLIST.contains(&struct_name)
    }
    // fn generate_define(
    //     define_name: &str,
    //     spec: &vk_parse::TypeCode,
    // ) -> Option<TokenStream> {
    //     let name = C::strip_constant_prefix(define_name)
    //         .unwrap_or(define_name);
    //     let ident = format_ident!("{}", name);

    //     if define_name.contains("VERSION") && !spec.code.contains("//#define") {
    //         let link = khronos_link(define_name);
    //         let (c_expr, (comment, (_name, parameters))) = parse_c_define_header(&spec.code).unwrap();
    //         let c_expr = c_expr.trim().trim_start_matches('\\');
    //         let c_expr = c_expr.replace("(uint32_t)", "");
    //         let c_expr = convert_c_expression::<C>(&c_expr, identifier_renames);
    //         let c_expr = discard_outmost_delimiter(c_expr);

    //         let deprecated = comment
    //             .and_then(|c| c.trim().strip_prefix("DEPRECATED: "))
    //             .map(|comment| quote!(#[deprecated = #comment]))
    //             .or_else(|| match define.deprecated.as_ref()?.as_str() {
    //                 "true" => Some(quote!(#[deprecated])),
    //                 "aliased" => {
    //                     Some(quote!(#[deprecated = "an old name not following Vulkan conventions"]))
    //                 }
    //                 x => panic!("Unknown deprecation reason {}", x),
    //             });

    //         let (code, ident) = if let Some(parameters) = parameters {
    //             let params = parameters
    //                 .iter()
    //                 .map(|param| format_ident!("{}", param))
    //                 .map(|i| quote!(#i: u32));
    //             let ident = format_ident!("{}", name.to_lowercase());
    //             (
    //                 quote!(pub const fn #ident(#(#params),*) -> u32 { #c_expr }),
    //                 ident,
    //             )
    //         } else {
    //             (quote!(pub const #ident: u32 = #c_expr;), ident)
    //         };

    //         identifier_renames.insert(define_name.clone(), ident);

    //         Some(quote! {
    //             #deprecated
    //             #[doc = #link]
    //             #code
    //         })
    //     } else {
    //         None
    //     }
    // }
    fn update_bindgen_header(header: &str, registry_path: &Path, bindings: bindgen::Builder) -> bindgen::Builder {
        let path = if header == "vk_platform.h" {
            // Fix broken path, https://github.com/KhronosGroup/Vulkan-Docs/pull/1538
            // Reintroduced in: https://github.com/KhronosGroup/Vulkan-Docs/issues/1573
            registry_path.join("vulkan").join(header)
        } else {
            registry_path.join(header)
        };
        bindings.header(path.to_str().expect("Valid UTF8 string"))
    }
}
