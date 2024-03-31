use proc_macro2::{
    Ident,
    TokenStream,
    Span,
};
use std::{
    borrow::Cow,
    path::Path,
};
use crate::util::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionType<'a> {
    Static,
    Entry,
    Dispatched(&'a str),
}

#[derive(Debug, Clone, Copy)]
pub struct FunctionsConfig<'a> {
    pub static_fns: &'a [&'a str],
    pub entry_fns: &'a [&'a str],
    pub dispatch_fns: &'a [(&'a str, &'a str)],
    pub dispatch_types: &'a [&'a str],
    pub dispatch_hints: &'a [(&'a str, &'a str)],
}

#[derive(Debug, Clone, Copy)]
pub struct ResultConfig<'a> {
    pub ty: &'a str,
    pub prefix: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub struct StructConfig<'a> {
    pub ty: &'a str,
    pub next_name: &'a str,
    pub ty_name: &'a str,
}

pub trait ApiConfig {
    const NAME: &'static str;
    const MODULE_NAME: &'static str;
    const REGISTRY_FILENAME: &'static str;
    const TYPE_PREFIX: &'static str;
    const COMMAND_PREFIX: &'static str;
    const CONSTANT_PREFIX: &'static str;
    const TAGGED_STRUCT: StructConfig<'static>;
    const RESULT_ENUM: ResultConfig<'static>;
    /// **TODO**: this should be read from the spec xml (see `tags`), but code organization of
    /// [`crate::variant_ident`] means we'd have to pass that contextual info though 100 places to
    /// get it in there
    ///
    /// <https://github.com/ash-rs/ash/blob/aee0c61cf1d466c6cf934f2375063fb88b806476/generator/src/lib.rs#L3374>
    const VENDOR_SUFFIXES: &'static [&'static str];
    const IGNORE_REQUIRED: bool = false;
    fn function_type(f: &vk_parse::CommandDefinition) -> FunctionType<'_>;
    /// based on original function name, gets the name of a the function pointer type
    ///
    /// e.x. `xrGetInstanceProcAddr` -> `PFN_xrGetInstanceProcAddr`
    fn function_pointer_name(fn_name: &str) -> Ident;
    /// return `true` to allow for separate setting of the length member of an array independently
    /// from it's data via a slice
    #[inline(always)]
    fn member_separate_count(_struct_name: &str, _member: &vk_parse::TypeMemberDefinition) -> bool {
        false
    }
    /// return `false` to disable creation of the builder pattern functions
    #[inline(always)]
    fn generate_accessors(_struct_name: &str) -> bool {
        true
    }
    /// use to override struct rendering on a per-struct basis
    ///
    /// default implementation returns `None` always
    #[inline(always)]
    fn manual_struct(_struct_name: &str) -> Option<TokenStream> {
        None
    }
    #[inline(always)]
    fn manual_derives(_struct_name: &str) -> Option<TokenStream> {
        None
    }
    /// Use to override conversion of type name to variant prefix string,
    ///
    /// i.e. `XrStructureType` -> `XR_TYPE_`
    fn variant_prefix(_type_name: &str) -> Option<Cow<'_, str>> {
        None
    }

    /// use to generate code for `#define`s, like [`crate::vulkan::Vulkan`] does for its
    /// `_VERSION_` macros
    #[inline(always)]
    fn generate_define(
        _name: &str,
        _spec: &vk_parse::TypeCode,
    ) -> Option<TokenStream> {
        None
    }

    fn update_bindgen_header(header: &str, _registry_path: &Path, bindings: bindgen::Builder) -> bindgen::Builder {
        bindings.header(header)
    }
}

pub trait ApiExt: ApiConfig {
    #[inline(always)]
    fn result_variant_prefix() -> &'static str {
        let default_suffix = Self::CONSTANT_PREFIX
            .strip_suffix('_')
            .unwrap_or(Self::CONSTANT_PREFIX);
        Self::RESULT_ENUM.prefix
            .unwrap_or(default_suffix)
    }

    fn name_to_tokens(type_name: &str) -> Ident {
        let new_name = match type_name {
            "uint8_t" => "u8",
            "uint16_t" => "u16",
            "uint32_t" => "u32",
            "uint64_t" => "u64",
            "int8_t" => "i8",
            "int16_t" => "i16",
            "int32_t" => "i32",
            "int64_t" => "i64",
            "size_t" => "usize",
            "int" => "c_int",
            "void" => "c_void",
            "char" => "c_char",
            "float" => "f32",
            "double" => "f64",
            "long" => "c_ulong",
            s => s.strip_prefix(Self::TYPE_PREFIX)
                .unwrap_or(s),
        };
        Ident::new(new_name.bits_to_flags().as_str(), Span::call_site())
    }

    #[inline(always)]
    fn strip_constant_prefix(s: &str) -> Option<&str> {
        s.strip_prefix(Self::CONSTANT_PREFIX)
    }

    #[inline(always)]
    fn strip_command_prefix(s: &str) -> Option<&str> {
        s.strip_prefix(Self::COMMAND_PREFIX)
    }

    #[inline(always)]
    fn strip_type_prefix(s: &str) -> Option<&str> {
        s.strip_prefix(Self::TYPE_PREFIX)
    }

    fn strip_vendor_suffix(name: &str) -> (&'static str, &str) {
        Self::VENDOR_SUFFIXES.iter().copied()
            .filter_map(|vendor| name.strip_vendor_suffix(vendor).map(move |n| (vendor, n)))
            .max_by_key(|(vendor, ..)| vendor.len())
            .unwrap_or(("", name))
    }

    #[inline(always)]
    fn contains_desired_api(api_list: &str) -> bool {
        api_list.split(',').any(|v| v == Self::NAME)
    }
}

impl<T: ApiConfig + ?Sized> ApiExt for T {}

impl<'a> FunctionsConfig<'a> {
    #[inline]
    fn dispatch_type<F>(&self, cmd: &vk_parse::CommandDefinition, strip_type: F) -> Option<&'a str>
    where
        F: for<'s> FnOnce(&'s str) -> &'s str,
    {
        let name = cmd.params.first()
            .and_then(|param| param.definition.type_name.as_ref())
            .map(AsRef::as_ref)
            .map(strip_type)?;
        self.dispatch_types.iter()
            .copied()
            .map(|ty| (ty, ty))
            .chain(self.dispatch_hints.iter().copied())
            .find(|&(alias, ..)| alias == name)
            .map(|(_alias, ty)| ty)
    }
    #[inline]
    fn known_types(&self) -> impl Iterator<Item=(&'a str, FunctionType<'a>)> {
        self.static_fns.iter()
            .copied()
            .map(|name| (name, FunctionType::Static))
            .chain({
                self.entry_fns.iter()
                    .copied()
                    .map(|name| (name, FunctionType::Entry))
            })
            .chain({
                self.dispatch_fns.iter()
                    .copied()
                    .map(|(name, ty)| (name,  FunctionType::Dispatched(ty)))
            })
    }
    #[inline]
    fn default_type(&self) -> Option<FunctionType<'a>> {
        self.dispatch_types.first()
            .copied()
            .map(FunctionType::Dispatched)
    }

    pub fn function_type<'b, F>(&self, cmd: &'b vk_parse::CommandDefinition, strip_type: F) -> Result<FunctionType<'a>, FunctionTypeError<'b>>
    where
        F: for<'s> FnOnce(&'s str) -> &'s str,
    {
        let name = cmd.proto.name.as_str();
        self.known_types()
            .find_map(|(n, ty)| if n == name {
                Some(ty)
            } else {
                None
            })
            .or_else(move || self.dispatch_type(cmd, strip_type).map(FunctionType::Dispatched))
            .or(self.default_type())
            .ok_or(FunctionTypeError(name))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("failed to find type for function: {0}")]
pub struct FunctionTypeError<'a>(&'a str);

// pub trait ApiConfig: Api {
//     fn desired_api(&self) -> &'static str;
//     fn contains_desired_api(&self, api_list: &str) -> bool {
//         let desired = self.desired_api();
//         api_list.split(',').any(|v| v == desired)
//     }
//     fn strip_bits_suffix(&self, s: &str) -> String;
//     fn bits_to_flags(&self, s: &str) -> String;
//     fn flags_to_bits(&self, s: &str) -> String;
//     fn without_type_prefix<'b>(&self, name: &'b str) -> Option<&'b str>;
//     #[inline(always)]
//     fn strip_type_prefix<'b>(&self, name: &'b str) -> &'b str {
//         self.without_type_prefix(name).unwrap_or(name)
//     }
//     fn without_command_prefix<'b>(&self, name: &'b str) -> Option<&'b str>;
//     #[inline(always)]
//     fn strip_command_prefix<'b>(&self, name: &'b str) -> &'b str {
//         self.without_command_prefix(name).unwrap_or(name)
//     }
//     fn without_constant_prefix<'b>(&self, name: &'b str) -> Option<&'b str>;
//     #[inline(always)]
//     fn constant_name<'b>(&self, name: &'b str) -> &'b str {
//         self.without_constant_prefix(name).unwrap_or(name)
//     }
//     fn name_is_bit(&self, name: &str) -> bool;
//     #[inline(always)]
//     fn constant_type_override(&self, _name: &str, _c: &Constant) -> Option<CType> {
//         None
//     }
//     fn vendor_suffixes(&self) -> &'static [&'static str];
//     fn is_result_type(&self, name: &str) -> bool;
//     fn result_variant_name<'b>(&self, name: &'b str) -> &'b str;
//     fn function_type(&self, cmd: &vk_parse::CommandDefinition) -> FunctionType<'static>;
//     fn is_structure_type(&self, field_ty: &str) -> bool;
//     fn pfn_prefix(&self) -> &'static str;
//     fn pfn_type_name(&self, fn_name: &str) -> syn::Ident {
//         format_ident!("{}{}", self.pfn_prefix(), fn_name)
//     }
//     fn allowed_count_members(&self) -> &'static [(&'static str, &'static str)];
//     fn is_next_member<T: PartialEq<str>>(&self, name: &T) -> bool;
//     fn is_type_member<T: PartialEq<str>>(&self, name: &T) -> bool;
//     fn generate_accessors(&self, s: &vkxml::Struct) -> bool;
//     fn struct_tag_type(&self) -> &'static str;
//     fn manual_derives(&self, s: &vkxml::Struct) -> TokenStream;
//     fn manual_structs(&self, s: &vkxml::Struct) -> Option<TokenStream>;
//     fn module_name(&self) -> &'static str;
//     /// list of types that are known to break the naming conventions of enum variants.
//     ///
//     /// example: `XrStructureType` only prefixes variants with `XR_TYPE_` instead of
//     /// `XR_STRUCTURE_TYPE_`
//     ///
//     /// 1. apply [`ApiConfig::bits_to_flags`]
//     /// 2. `SCREAMING_SNAKE_CASE`
//     /// 3. remove vendor suffixes if present
//     /// 4. correct number suffix spacing
//     fn override_variant_prefix<'a>(&'a self, _type_name: &str) -> Option<&'a str> {
//         None
//     }
// }

// impl ApiConfig for VkConfig<'static> {
//     fn desired_api(&self) -> &'static str {
//         "vulkan"
//     }
//     fn strip_bits_suffix(&self, s: &str) -> String {
//         s.replace(self.flag_suffixes.0, "")
//     }
//     fn bits_to_flags(&self, s: &str) -> String {
//         let (bits_suffix, flags_suffix) = self.flag_suffixes;
//         s.replace(bits_suffix, flags_suffix)
//     }
//     fn flags_to_bits(&self, s: &str) -> String {
//         let (bits_suffix, flags_suffix) = self.flag_suffixes;
//         s.replace(flags_suffix, bits_suffix)
//     }
//     #[inline(always)]
//     fn without_type_prefix<'b>(&self, name: &'b str) -> Option<&'b str> {
//         name.strip_prefix(self.type_prefix)
//     }
//     #[inline(always)]
//     fn without_command_prefix<'b>(&self, name: &'b str) -> Option<&'b str> {
//         name.strip_prefix(self.command_prefix)
//     }
//     fn without_constant_prefix<'b>(&self, name: &'b str) -> Option<&'b str> {
//         name.strip_prefix(self.constant_prefix)
//     }
//     #[inline(always)]
//     fn name_is_bit(&self, name: &str) -> bool {
//         name.contains("Bit")
//     }
//     fn constant_type_override(&self, name: &str, _c: &Constant) -> Option<CType> {
//         if name == "TRUE" || name == "FALSE" {
//             Some(CType::Bool32)
//         } else {
//             None
//         }
//     }
//     #[inline(always)]
//     fn vendor_suffixes(&self) -> &'static [&'static str] {
//         self.vendor_suffixes
//     }
//     #[inline(always)]
//     fn is_result_type(&self, name: &str) -> bool {
//         name == self.result.ty
//     }
//     #[inline(always)]
//     fn result_variant_name<'b>(&self, name: &'b str) -> &'b str {
//         name.strip_prefix(self.result.prefix)
//             .expect("result variant did not have proper prefix")
//     }
//     fn function_type(&self, cmd: &vk_parse::CommandDefinition) -> FunctionType<'static> {
//         self.functions.function_type(cmd, |s| self.strip_type_prefix(s))
//             .expect("failed to get function type")
//     }
//     fn is_structure_type(&self, field_ty: &str) -> bool {
//         self.tagged_struct.ty == field_ty
//     }
//     #[inline(always)]
//     fn pfn_prefix(&self) -> &'static str {
//         self.pfn_prefix.unwrap_or("PFN_")
//     }
//     #[inline(always)]
//     fn allowed_count_members(&self) -> &'static [(&'static str, &'static str)] {
//         self.allowed_count_members
//     }
//     fn is_next_member<T: PartialEq<str>>(&self, name: &T) -> bool {
//         name == self.tagged_struct.next_name
//     }
//     fn is_type_member<T: PartialEq<str>>(&self, name: &T) -> bool {
//         name == self.tagged_struct.ty_name
//     }
//     fn generate_accessors(&self, s: &vkxml::Struct) -> bool {
//         let name = s.name.as_str();
//         !self.accessor_blacklist.iter().copied()
//             .any(|v| v == name)
//     }
//     #[inline(always)]
//     fn struct_tag_type(&self) -> &'static str {
//         self.tagged_struct.ty
//     }
//     /// At the moment `Ash` doesn't properly derive all the necessary drives
//     /// like Eq, Hash etc.
//     /// To Address some cases, you can add the name of the struct that you
//     /// require and add the missing derives yourself.
//     fn manual_derives(&self, s: &vkxml::Struct) -> TokenStream {
//         match s.name.as_str() {
//             "VkClearRect" | "VkExtent2D" | "VkExtent3D" | "VkOffset2D" | "VkOffset3D" | "VkRect2D"
//             | "VkSurfaceFormatKHR" => quote! {PartialEq, Eq, Hash,},
//             _ => quote! {},
//         }
//     }
//     fn manual_structs(&self, s: &vkxml::Struct) -> Option<TokenStream> {
//         match s.name.as_str() {
//             "VkTransformMatrixKHR" => {
//                 Some(quote! {
//                     #[repr(C)]
//                     #[derive(Copy, Clone)]
//                     pub struct TransformMatrixKHR {
//                         pub matrix: [f32; 12],
//                     }
//                 })
//             },
//             "VkAccelerationStructureInstanceKHR" => {
//                 Some(quote! {
//                     #[repr(C)]
//                     #[derive(Copy, Clone)]
//                     pub union AccelerationStructureReferenceKHR {
//                         pub device_handle: DeviceAddress,
//                         pub host_handle: AccelerationStructureKHR,
//                     }
//                     #[repr(C)]
//                     #[derive(Copy, Clone)]
//                     #[doc = "<https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureInstanceKHR.html>"]
//                     pub struct AccelerationStructureInstanceKHR {
//                         pub transform: TransformMatrixKHR,
//                         /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
//                         pub instance_custom_index_and_mask: Packed24_8,
//                         /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
//                         pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
//                         pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
//                     }
//                 })
//             },
//             "VkAccelerationStructureSRTMotionInstanceNV" => {
//                 Some(quote! {
//                     #[repr(C)]
//                     #[derive(Copy, Clone)]
//                     #[doc = "<https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAccelerationStructureSRTMotionInstanceNV.html>"]
//                     pub struct AccelerationStructureSRTMotionInstanceNV {
//                         pub transform_t0: SRTDataNV,
//                         pub transform_t1: SRTDataNV,
//                         /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
//                         pub instance_custom_index_and_mask: Packed24_8,
//                         /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
//                         pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
//                         pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
//                     }
//                 })
//             },
//             "VkAccelerationStructureMatrixMotionInstanceNV" => {
//                 Some(quote! {
//                     #[repr(C)]
//                     #[derive(Copy, Clone)]
//                     #[doc = "<https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/AccelerationStructureMatrixMotionInstanceNV.html>"]
//                     pub struct AccelerationStructureMatrixMotionInstanceNV {
//                         pub transform_t0: TransformMatrixKHR,
//                         pub transform_t1: TransformMatrixKHR,
//                         /// Use [`Packed24_8::new(instance_custom_index, mask)`][Packed24_8::new()] to construct this field
//                         pub instance_custom_index_and_mask: Packed24_8,
//                         /// Use [`Packed24_8::new(instance_shader_binding_table_record_offset, flags)`][Packed24_8::new()] to construct this field
//                         pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,
//                         pub acceleration_structure_reference: AccelerationStructureReferenceKHR,
//                     }
//                 })
//             },
//             _ => None,
//         }
//     }
//     #[inline(always)]
//     fn module_name(&self) -> &'static str {
//         self.module_name
//     }
//     fn override_variant_prefix<'a>(&'a self, type_name: &str) -> Option<&'a str> {
//         self.variant_prefixes.iter().copied()
//             .find(|&(k, ..)| k == type_name)
//             .map(|(_k, v)| v)
//     }
// }
// 
// #[derive(Debug, Clone, Copy)]
// pub struct VkConfig<'a> {
//     pub desired_api: &'a str,
//     pub type_prefix: &'a str,
//     pub command_prefix: &'a str,
//     pub constant_prefix: &'a str,
//     pub flag_suffixes: (&'a str, &'a str),
//     pub vendor_suffixes: &'a [&'a str],
//     pub result: ResultConfig<'a>,
//     pub functions: FunctionsConfig<'a>,
//     pub pfn_prefix: Option<&'a str>,
//     pub allowed_count_members: &'a [(&'a str, &'a str)],
//     pub accessor_blacklist: &'a [&'a str],
//     pub tagged_struct: StructConfig<'a>,
//     pub module_name: &'a str,
//     pub registry_filename: &'a str,
//     pub variant_prefixes: &'a [(&'a str, &'a str)],
// }
