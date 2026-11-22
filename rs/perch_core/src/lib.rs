pub mod frames;
pub mod model;
pub mod perception;
pub mod phase;
pub mod q2;

pub const G: f32 = 9.81;
pub type Action = [f32; 4];

pub(crate) fn invalid(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}
