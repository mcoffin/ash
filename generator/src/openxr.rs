use proc_macro2::{
    TokenStream,
    Ident,
};
use quote::{
    quote,
    format_ident,
};
use std::{
    path::Path,
    borrow::Cow
};
use crate::get_variant;
use crate::config::{
    self,
    ApiConfig,
    FunctionType,
};
use super::{
    CommandMap,
    ConstantTypeInfo,
    EnumType,
};
use regex::Regex;
use once_cell::sync::Lazy;

#[derive(Debug, Clone, Copy)]
pub struct OpenXR;

impl OpenXR {
    const FUNCTIONS: config::FunctionsConfig<'static> = config::FunctionsConfig {
        static_fns: &[
            "xrGetInstanceProcAddr",
        ],
        entry_fns: &[
            "xrCreateInstance",
            "xrEnumerateInstanceLayerProperties",
            "xrEnumerateInstanceExtensionProperties",
        ],
        dispatch_fns: &[],
        dispatch_types: &[
            "Instance",
        ],
        dispatch_hints: &[],
    };
}

macro_rules! xr_vendor_suffixes {
    ($($v:expr),+ $(,)?) => {
        &[
            $(
                $v,
            )+
        ]
    };
}

impl ApiConfig for OpenXR {
    const NAME: &'static str = "openxr";
    const MODULE_NAME: &'static str = "xr";
    const REGISTRY_FILENAME: &'static str = "xr.xml";
    const TYPE_PREFIX: &'static str = "Xr";
    const COMMAND_PREFIX: &'static str = "xr";
    const CONSTANT_PREFIX: &'static str = "XR_";
    const TAGGED_STRUCT: config::StructConfig<'static> = config::StructConfig {
        ty: "XrStructureType",
        ty_name: "type",
        next_name: "next",
    };
    const RESULT_ENUM: config::ResultConfig<'static> = config::ResultConfig {
        ty: "XrResult",
        prefix: None,
    };
    const VENDOR_SUFFIXES: &'static [&'static str] = xr_vendor_suffixes!{
        "ACER",
        "ALMALENCE",
        "ANDROID",
        "ANDROIDSYS",
        "ARM",
        "BD",
        "COLLABORA",
        "DANWILLM",
        "EPIC",
        "EXT",
        "EXTX",
        "FB",
        "FREDEMMOTT",
        "GOOGLE",
        "HTC",
        "HUAWEI",
        "INTEL",
        "KHR",
        "LUNARG",
        "LIV",
        "LOGITECH",
        "META",
        "ML",
        "MND",
        "MNDX",
        "MSFT",
        "NV",
        "OCULUS",
        "OPPO",
        "PLUTO",
        "QCOM",
        "STARBREEZE",
        "TOBII",
        "ULTRALEAP",
        "UNITY",
        "VALVE",
        "VARJO",
        "YVR",
    };
    const IGNORE_REQUIRED: bool = true;
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
    fn variant_prefix<'a>(type_name: &'a str) -> Option<Cow<'a, str>> {
        match type_name {
            "XrStructureType" => Some(Cow::from("XR_TYPE")),
            "XrPerfSettingsNotificationLevelEXT" => Some(Cow::from("XR_PERF_SETTINGS_NOTIF_LEVEL")),
            "XrLoaderInterfaceStructs" => Some(Cow::from("XR_LOADER_INTERFACE_STRUCT")),
            _ => None,
        }
    }
    fn update_bindgen_header(header: &str, _registry_path: &Path, bindings: bindgen::Builder) -> bindgen::Builder {
        static OXR_HEADER: Lazy<Regex> = Lazy::new(|| Regex::new(r"^openxr.*\.h$").unwrap());
        let path = if OXR_HEADER.is_match_at(header, 0) {
            Cow::Owned(format!("/usr/include/openxr/{}", header))
        } else {
            Cow::Borrowed(header)
        };
        bindings.header(path)
    }
    fn manual_derives(name: &str) -> Option<TokenStream> {
        static NORMAL_STRUCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"^Xr((Extent)|(Rect)|(Offset))[0-9]D$").unwrap());
        if NORMAL_STRUCT.is_match(name) {
            Some(quote!(PartialEq, Eq, Hash))
        } else {
            None
        }
    }
}
