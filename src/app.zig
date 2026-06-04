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
