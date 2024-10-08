mod aliases;
pub use aliases::*;
mod bitflags {
    use ash::vk::Flags;
    include!("./xr/bitflags.rs");
}
pub use bitflags::*;
mod constants;
pub use constants::*;
#[cfg(feature = "debug")]
mod const_debugs;
mod definitions;
pub use definitions::*;
mod enums;
pub use enums::*;
mod extensions;
pub use extensions::*;
mod feature_extensions;
pub use feature_extensions::*;
mod features;
pub use features::*;
#[path = "./xr_prelude.rs"]
mod prelude;
pub use prelude::*;
/// Native bindings from Vulkan headers, generated by bindgen
#[allow(clippy::useless_transmute, nonstandard_style)]
mod native;
pub use native::*;
#[path = "./xr_platform_types.rs"]
mod platform_types;
pub use platform_types::*;

macro_rules! chain_iter {
    ($name:ident : $t:ident, $next:ident) => {
        /// Iterates through the pointer chain. Includes the item that is passed into the function.
        /// Stops at the last [`BaseOutStructure`] that has a null [`BaseOutStructure::p_next`] field.
        pub(crate) unsafe fn $name<T: ?Sized>(
            ptr: &mut T,
        ) -> impl Iterator<Item = *mut $t<'_>> {
            let ptr = <*mut T>::cast::<$t<'_>>(ptr);
            (0..).scan(ptr, |p_ptr, _| {
                if p_ptr.is_null() {
                    return None;
                }
                let n_ptr = (**p_ptr).$next;
                let old = *p_ptr;
                *p_ptr = n_ptr;
                Some(old)
            })
        }
    };
}
chain_iter!(ptr_chain_iter : BaseOutStructure, next);

pub use ash::vk::Handle;