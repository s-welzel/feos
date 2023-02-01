use quote::quote;
use syn::DeriveInput;

pub(crate) fn expand_ideal_gas(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let variants = match input.data {
        syn::Data::Enum(syn::DataEnum { ref variants, .. }) => variants,
        _ => panic!("this derive macro only works on enums"),
    };

    let ideal_gas = impl_ideal_gas(variants);
    // let entropy_scaling = impl_entropy_scaling(variants)?;
    Ok(quote! {
        #ideal_gas
    })
}

fn impl_ideal_gas(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> proc_macro2::TokenStream {
    let components = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(ideal_gas) => ideal_gas.components()
        }
    });
    let subset = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(ideal_gas) => Self::#name(ideal_gas.subset(component_list))
        }
    });
    let de_broglie_wavelength = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(ideal_gas) => ideal_gas.de_broglie_wavelength()
        }
    });
    let display = variants.iter().map(|v| {
        let name = &v.ident;
        quote! {
            Self::#name(ideal_gas) => write!(f, "{}", ideal_gas.to_string())
        }
    }); 

    quote! {
        impl IdealGas for IdealGasModel {
            fn components(&self) -> usize {
                match self {
                    #(#components,)*
                }
            }
            fn subset(&self, component_list: &[usize]) -> Self {
                match self {
                    #(#subset,)*
                }
            }
            fn de_broglie_wavelength(&self) -> &Box<dyn DeBroglieWavelength> {
                match self {
                    #(#de_broglie_wavelength,)*
                }
            }
        }

        impl fmt::Display for IdealGasModel {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    #(#display,)*
                }
            }
        }
    }
}