use proc_macro2::{
    TokenStream,
    Ident,
};
use quote::{
    quote,
    format_ident,
};
use std::{
    collections::BTreeMap,
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
            "XrFaceExpression2FB" => Some(Cow::from("XR_FACE_EXPRESSION2")),
            "XrFaceExpressionSet2FB" => Some(Cow::from("XR_FACE_EXPRESSION_SET2")),
            "XrFaceTrackingDataSource2FB" => Some(Cow::from("XR_FACE_TRACKING_DATA_SOURCE2")),
            "XrFaceConfidence2FB" => Some(Cow::from("XR_FACE_CONFIDENCE2")),
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
    fn generate_define(
        define_name: &str,
        spec: &vk_parse::TypeCode,
        deprecated: Option<&str>,
        identifier_renames: &mut BTreeMap<String, Ident>,
    ) -> Option<TokenStream> {
        use super::*;
        use regex::Regex;
        use once_cell::sync::Lazy;
        static TYPE_CASTS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\(uint[0-9]{2}_t\)").unwrap());
        static HEX_SUFFIXES: Lazy<Regex> = Lazy::new(|| Regex::new(r"(0x[A-Fa-f0-9]+)UL*").unwrap());
        let name = Self::strip_constant_prefix(define_name)
            .unwrap_or(define_name);
        let ident = format_ident!("{}", name);

        if define_name.contains("VERSION") && !spec.code.contains("//#define") {
            let link = khronos_link(define_name);
            let code = &spec.code;
            let (c_expr, (comment, (_name, parameters))) = parse_c_define_header(code.trim_left()).unwrap();
            let c_expr = c_expr.trim().trim_start_matches('\\');
            let c_expr = TYPE_CASTS.replace_all(c_expr, "");
            let c_expr = HEX_SUFFIXES.replace_all(&*c_expr, "$1");
            let c_expr = convert_c_expression::<Self>(&c_expr, identifier_renames);
            let c_expr = discard_outmost_delimiter(c_expr);

            let deprecated = comment
                .and_then(|c| c.trim().strip_prefix("DEPRECATED: "))
                .map(|comment| quote!(#[deprecated = #comment]))
                .or_else(|| match deprecated? {
                    "true" => Some(quote!(#[deprecated])),
                    "aliased" => {
                        Some(quote!(#[deprecated = "an old name not following conventions"]))
                    }
                    x => panic!("Unknown deprecation reason {}", x),
                });

            let (code, ident) = if let Some(parameters) = parameters {
                let params = parameters
                    .iter()
                    .map(|param| format_ident!("{}", param))
                    .map(|i| quote!(#i: u64));
                let ident = format_ident!("{}", name.to_lowercase());
                (
                    quote!(pub const fn #ident(#(#params),*) -> u64 { #c_expr }),
                    ident,
                )
            } else {
                (quote!(pub const #ident: u32 = #c_expr;), ident)
            };

            identifier_renames.insert(define_name.to_owned(), ident);

            Some(quote! {
                #deprecated
                #[doc = #link]
                #code
            })
        } else {
            None
        }
    }
}
