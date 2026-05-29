const log = @import("std").log;

pub fn info(comptime msg: []const u8, args: anytype) void {
    log.info(msg, args);
}

pub fn debug(comptime msg: []const u8, args: anytype) void {
    log.debug(msg, args);
}

pub fn warn(comptime msg: []const u8, args: anytype) void {
    log.warn(msg, args);
}

pub fn err(comptime msg: []const u8, args: anytype) void {
    log.err(msg, args);
}
