use proc_macro2::TokenStream;
use std::{
    cmp::PartialEq,
    collections::{
        BTreeMap,
        HashMap,
        HashSet,
    },
    error::Error,
    iter,
    path::{
        Path,
        PathBuf,
    },
};
use crate::get_variant;
use crate::config::{
    self,
    ApiConfig,
    VkConfig,
};
use super::{
    CommandMap,
    ConstantTypeInfo,
    EnumType,
};

pub const XR_CONFIG: VkConfig<'static> = VkConfig {
    desired_api: "openxr",
    module_name: "xr",
    type_prefix: "Xr",
    command_prefix: "xr",
    constant_prefix: "XR_",
    registry_filename: "xr.xml",
    flag_suffixes: ("FlagBits", "Flags"),
    result: config::ResultConfig {
        ty: "XrResult",
        prefix: "XR",
    },
    functions: config::FunctionsConfig {
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
        dispatch_aliases: &[],
    },
    tagged_struct: config::StructConfig {
        ty: stringify!(XrStructureType),
        next_name: "next",
        ty_name: "type",
    },
    ..super::VK_CONFIG
};

pub fn write_source_code<
    RegP: Into<PathBuf>,
    OutP: AsRef<Path>
>(
    xr_registry: RegP,
    src_dir: OutP,
) -> Result<(), Box<dyn Error>> {
    let mut xr_xml = xr_registry.into();
    xr_xml.extend(iter::once("vk.xml"));
    debug!("xr_xml: {}", xr_xml.display());
    debug!("src_dir: {}", src_dir.as_ref().display());
    let (spec, spec2) = parse_registry(xr_xml)
        .or_else(|e| match e {
            RegistryParseError::NonFatal { value, errors } => {
                eprintln!("non-fatal registry parse errors: {:?}", &*errors);
                Ok(value)
            },
            e => Err(e),
        })?;
    debug!("registry: {:?}", &spec2);
    let extensions = spec2.supported_extensions(&XR_CONFIG)?
        .collect::<Vec<_>>();
    debug!("extensions: {:#?}", &extensions);
    let features: Vec<_> = spec.features().collect();
    debug!("features: {:#?}", features.as_slice());
    let definitions: Vec<_> = spec.definitions().collect();
    debug!("definitions: {:#?}", definitions.as_slice());
    let constants: Vec<_> = spec.constants().collect();
    debug!("constants: {:#?}", constants.as_slice());
    let extension_children = extensions.iter()
        .flat_map(|ext| &ext.children);
    let reqs = spec2.features_children(&XR_CONFIG)
        .chain(extensions.iter().flat_map(|ext| &ext.children))
        .filter_map(|v| match v {
            vk_parse::FeatureChild::Require { api, items, .. } => match api.as_deref() {
                None => Some(items),
                Some(api) if api == XR_CONFIG.desired_api() => Some(items),
                _ => None,
            },
            _ => None,
        })
        .flat_map(|items| items)
        .collect::<Requirements<'_>>();
    debug!("requirements: {:#?}", &reqs);
    let commands: CommandMap<'_> = spec2.command_definitions()
        .filter_map(|cmd @ &vk_parse::CommandDefinition { proto: vk_parse::NameWithType { ref name, .. }, .. }| {
            if reqs.commands.contains(name.as_str()) {
                Some((name.clone(), cmd))
            } else {
                None
            }
        })
        .collect();
    let cmd_aliases: HashMap<&'_ str, &'_ str> = spec2.command_aliases()
        .filter(|(name, _alias)| reqs.commands.contains(name.as_str()))
        .map(|(name, alias)| (name.as_ref(), alias.as_ref()))
        .collect();

    let mut cache = Cache::new(&XR_CONFIG);

    let EnumCode {
        enums: enum_code,
        bitflags: bitflags_code
    } = spec2.enums()
        .filter(|es| es.kind.is_some())
        .filter(|es| es.name.as_ref().map_or(true, |n| reqs.contains_flags(n.as_ref(), &XR_CONFIG)))
        .map(|e| cache.generate_enum(e))
        .collect();

    let mut constants_code: Vec<_> = constants.iter()
        .map(|constant| cache.generate_constant(constant))
        .collect();

    let union_types: HashSet<_> = definitions.iter()
        .filter_map(|v| match v {
            vkxml::DefinitionsElement::Union(vkxml::Union { name, .. }) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    let mut identifier_renames: BTreeMap<String, syn::Ident> = BTreeMap::new();
    let mut has_lifetimes = definitions.iter()
        .filter_map(get_variant!(vkxml::DefinitionsElement::Struct))
        .filter(|s| s.members().any(|vkxml::Field { reference, .. }| reference.is_some()))
        .map(|vkxml::Struct { name, .. }| XR_CONFIG.name_to_tokens(name.as_ref()))
        .collect::<HashSet<_>>();
    definitions.iter().for_each(|v| {
        let name = match v {
            vkxml::DefinitionsElement::Struct(s) => {
                if s.members().any(|field| has_lifetimes.contains(&XR_CONFIG.name_to_tokens(&field.basetype))) {
                    Some(s.name.as_str())
                } else {
                    None
                }
            },
            vkxml::DefinitionsElement::Union(u) => {
                u.elements.iter()
                    .find(|field| has_lifetimes.contains(&XR_CONFIG.name_to_tokens(&field.basetype)))
                    .map(|_| u.name.as_str())
            },
            _ => None,
        };
        if let Some(name) = name {
            has_lifetimes.insert(XR_CONFIG.name_to_tokens(name));
        }
    });
    spec2.types()
        .filter_map(|v| match v {
            vk_parse::Type { name: Some(name), alias: Some(alias), .. } => Some((name, alias)),
            _ => None
        })
        .for_each(|(name, alias)| {
            if has_lifetimes.contains(&XR_CONFIG.name_to_tokens(alias)) {
                has_lifetimes.insert(XR_CONFIG.name_to_tokens(name));
            }
        });
    unimplemented!();
}

type Registry = (vkxml::Registry, vk_parse::Registry);
fn parse_registry<P: AsRef<Path>>(
    path: P
) -> Result<Registry, RegistryParseError<Registry>> {
    let path = path.as_ref();
    let parse_vkxml = || {
        vk_parse::parse_file_as_vkxml(path)
            .map_err(RegistryParseError::VkXml)
    };
    let spec2 = vk_parse::parse_file(path)
        .map_err(RegistryParseError::Fatal)
        .and_then(|(reg, errors)| if errors.is_empty() {
            Ok(reg)
        } else {
            parse_vkxml().and_then(move |spec| Err(RegistryParseError::NonFatal {
                value: (spec, reg),
                errors: errors.into(),
            }))
        })?;
    let spec = parse_vkxml()?;
    Ok((spec, spec2))
}

#[derive(Debug, thiserror::Error)]
enum RegistryParseError<T> {
    #[error("error parsing registry: {0:?}")]
    Fatal(vk_parse::FatalError),
    #[error("error parsing registry as vkxml: {0:?}")]
    VkXml(vk_parse::FatalError),
    #[error("non-fatal errors while parsing registry: {errors:?}")]
    NonFatal {
        value: T,
        errors: Box<[vk_parse::Error]>,
    },
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("failed to find extensions child in registry")]
struct ExtensionsFindError;

trait IteratorExt: Iterator + Sized {
    #[inline(always)]
    fn contains<T>(mut self, value: T) -> bool
    where
        <Self as Iterator>::Item: PartialEq<T>,
    {
        self.any(move |v| v == value)
    }
}

impl<It> IteratorExt for It
where
    It: Iterator,
{}

trait RegistryExt {
    fn supported_extensions<'a, C: ApiConfig + ?Sized>(&'a self, config: &C) -> Result<impl Iterator<Item=&'a vk_parse::Extension>, ExtensionsFindError>;
    fn supported_features<'a, C: ApiConfig + ?Sized>(&'a self, config: &C) -> impl Iterator<Item=&'a vk_parse::Feature>;
    fn features_children<'a, C>(&'a self, config: &C) -> impl Iterator<Item=&'a vk_parse::FeatureChild>
    where
        C: ApiConfig + ?Sized,
    {
        self.supported_features(config)
            .flat_map(|vk_parse::Feature { children, .. }| children.as_slice())
    }
    fn commands<'a>(&'a self) -> impl Iterator<Item=&'a vk_parse::Command>;
    fn command_definitions<'a>(&'a self) -> impl Iterator<Item=&'a vk_parse::CommandDefinition> {
        self.commands()
            .filter_map(|v| match v {
                vk_parse::Command::Definition(v) => Some(v),
                _ => None,
            })
    }
    fn command_aliases<'a>(&'a self) -> impl Iterator<Item=(&'a String, &'a String)> {
        self.commands()
            .filter_map(|v| match v {
                vk_parse::Command::Alias { name, alias } => Some((name, alias)),
                _ => None,
            })
    }
    fn enums<'a>(&'a self) -> impl Iterator<Item=&'a vk_parse::Enums>;
    fn types<'a>(&'a self) -> impl Iterator<Item=&'a vk_parse::Type>;
}

impl RegistryExt for vk_parse::Registry {
    fn supported_extensions<'a, C: ApiConfig + ?Sized>(&'a self, config: &C) -> Result<impl Iterator<Item=&'a vk_parse::Extension>, ExtensionsFindError> {
        use vk_parse::RegistryChild;
        let extensions = self.0.iter()
            .find_map(|v| match v {
                RegistryChild::Extensions(v) => Some(v),
                _ => None,
            })
            .ok_or(ExtensionsFindError)?;
        let ret = extensions.children.iter()
            .filter(|&ext| ext.is_supported(config.desired_api()));
        Ok(ret)
    }
    fn supported_features<'a, C: ApiConfig + ?Sized>(&'a self, config: &C) -> impl Iterator<Item=&'a vk_parse::Feature> {
        self.0.iter()
            .filter_map(|v| match v {
                vk_parse::RegistryChild::Feature(v) if v.is_supported(config.desired_api()) => Some(v),
                _ => None,
            })
    }
    fn commands<'a>(&'a self) -> impl Iterator<Item=&'a vk_parse::Command> {
        self.0.iter()
            .filter_map(|v| match v {
                vk_parse::RegistryChild::Commands(v) => Some(v),
                _ => None,
            })
            .flat_map(|cmds| &cmds.children)
    }
    fn enums<'a>(&'a self) -> impl Iterator<Item=&'a vk_parse::Enums> {
        self.0.iter().filter_map(|v| match v {
            vk_parse::RegistryChild::Enums(v) => Some(v),
            _ => None,
        })
    }
    fn types<'a>(&'a self) -> impl Iterator<Item=&'a vk_parse::Type> {
        self.0.iter()
            .filter_map(get_variant!(vk_parse::RegistryChild::Types))
            .flat_map(|types| &types.children)
            .filter_map(get_variant!(vk_parse::TypesChild::Type))
    }
}

macro_rules! xml_variant_getters {
    (trait $trait_name:ident { $($name:ident: $t:ty = $v:ident),+ $(,)? }) => {
        trait $trait_name {
            $(
                fn $name<'a>(&'a self) -> impl Iterator<Item=&'a $t>;
            )+
        }
        impl $trait_name for ::vkxml::Registry {
            $(
                fn $name<'a>(&'a self) -> impl Iterator<Item=&'a $t> {
                    self.elements.iter()
                        .filter_map(|v| match v {
                            ::vkxml::RegistryElement::$v(v) => Some(v),
                            _ => None,
                        })
                        .flat_map(|v| &v.elements)
                }
            )+
        }
    };
}

xml_variant_getters!(trait XmlRegistryExt {
    features: vkxml::Feature = Features,
    definitions: vkxml::DefinitionsElement = Definitions,
    constants: vkxml::Constant = Constants,
});

trait Supported {
    fn supported_apis<'a>(&'a self) -> Option<impl Iterator<Item=&'a str>>;
    fn is_supported(&self, api: &str) -> bool {
        self.supported_apis()
            .map_or(true, |apis| apis.contains(api))
    }
}

impl Supported for vk_parse::Extension {
    fn supported_apis<'a>(&'a self) -> Option<impl Iterator<Item=&'a str>> {
        self.supported.as_ref()
            .map(|s| s.split(','))
    }
}

impl Supported for vk_parse::Feature {
    fn supported_apis<'a>(&'a self) -> Option<impl Iterator<Item=&'a str>> {
        Some(self.api.split(','))
    }
}

#[derive(Debug)]
struct Requirements<'a> {
    types: HashSet<&'a str>,
    commands: HashSet<&'a str>,
}

impl<'a> Requirements<'a> {
    fn contains_flags<C: ApiConfig + ?Sized>(&self, name: &str, config: &C) -> bool {
        let name = config.bits_to_flags(name);
        self.types.contains(name.as_str())
    }
}

impl<'a> FromIterator<&'a vk_parse::InterfaceItem> for Requirements<'a> {
    fn from_iter<It>(it: It) -> Self
    where
        It: IntoIterator<Item=&'a vk_parse::InterfaceItem>,
    {
        use vk_parse::InterfaceItem;
        let mut types = HashSet::new();
        let mut commands = HashSet::new();
        it.into_iter().for_each(|elem| match elem {
            InterfaceItem::Type { name, .. } => {
                types.insert(name.as_str());
            },
            InterfaceItem::Command { name, .. } => {
                commands.insert(name.as_str());
            },
            _ => {},
        });
        Requirements {
            types,
            commands,
        }
    }
}

struct EnumCode {
    enums: Vec<TokenStream>,
    bitflags: Vec<TokenStream>,
}

impl FromIterator<EnumType> for EnumCode {
    fn from_iter<It>(it: It) -> Self
    where
        It: IntoIterator<Item=EnumType>,
    {
        let mut ret = EnumCode {
            enums: Vec::new(),
            bitflags: Vec::new(),
        };
        let &mut EnumCode { ref mut enums, ref mut bitflags } = &mut ret;
        it.into_iter().for_each(|elem| match elem {
            EnumType::Enum(v) => enums.push(v),
            EnumType::Bitflags(v) => enums.push(v),
        });
        ret
    }
}

trait StructExt {
    fn members<'a>(&'a self) -> impl Iterator<Item=&'a vkxml::Field>;
}
impl StructExt for vkxml::Struct {
    fn members<'a>(&'a self) -> impl Iterator<Item=&'a vkxml::Field> {
        self.elements.iter()
            .filter_map(get_variant!(vkxml::StructElement::Member))
    }
}

struct Cache<'a, C> {
    functions: HashMap<&'a str, (syn::Ident, syn::Ident)>,
    bitflags: HashSet<syn::Ident>,
    constants: HashSet<&'a str>,
    constant_values: BTreeMap<syn::Ident, ConstantTypeInfo>,
    config: &'a C,
}

impl<'a, C> Cache<'a, C>
where
    C: ApiConfig,
{
    #[inline(always)]
    fn new(config: &'a C) -> Self {
        Cache {
            functions: HashMap::new(),
            bitflags: HashSet::new(),
            constants: HashSet::new(),
            constant_values: BTreeMap::new(),
            config,
        }
    }

    fn generate_enum(
        &mut self,
        e: &'a vk_parse::Enums,
    ) -> EnumType {
        use super::generate_enum;
        generate_enum(e, &mut self.constants, &mut self.constant_values, &mut self.bitflags, self.config)
    }

    fn generate_constant(
        &mut self,
        constant: &'a vkxml::Constant,
    ) -> TokenStream {
        use super::generate_constant;
        generate_constant(constant, &mut self.constants, self.config)
    }

    fn generate_extension_constants(
        &mut self,
        extension_name: &str,
        extension_number: i64,
        extension_items: &'a [vk_parse::ExtensionChild],
    ) -> TokenStream {
        unimplemented!()
    }
}
