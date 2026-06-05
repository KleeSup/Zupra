const sapp = @import("sokol").app;

/// Request the application to quit. This will trigger the deinit callback and then exit the application.
pub fn quit() void {
    sapp.requestQuit();
}

/// Immediately exit the application without triggering the deinit callback.
/// Use with caution, as this will not clean up any resources!
pub fn exit() void {
    sapp.quit();
}

pub fn width() i32 {
    return sapp.width();
}

pub fn height() i32 {
    return sapp.height();
}

pub fn widthf() f32 {
    return sapp.widthf();
}

pub fn heightf() f32 {
    return sapp.heightf();
}

pub fn getDelta() f32 {
    return @as(f32, @floatCast(sapp.frameDuration()));
}
