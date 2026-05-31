pub const zm = @import("zmath");
const math = @import("std").math;

pub const DEG_TO_RAD = math.pi / 180.0;
pub const Matrix = zm.Mat;
pub const Quaternion = zm.Quat;

pub const Rect = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,

    pub fn zero() Rect {
        return .{ .x = 0, .y = 0, .width = 0, .height = 0 };
    }

    pub fn fromMinMax(min: Vec2, max: Vec2) Rect {
        return .{ .x = min.x, .y = min.y, .width = max.x - min.x, .height = max.y - min.y };
    }

    pub fn fromCenterSize(center: Vec2, size: Vec2) Rect {
        return .{ .x = center.x - size.x * 0.5, .y = center.y - size.y * 0.5, .width = size.x, .height = size.y };
    }

    pub fn getMin(self: Rect) Vec2 {
        return .{ .x = self.x, .y = self.y };
    }

    pub fn getMax(self: Rect) Vec2 {
        return .{ .x = self.x + self.width, .y = self.y + self.height };
    }

    pub fn getCenter(self: Rect) Vec2 {
        return .{ .x = self.x + self.width * 0.5, .y = self.y + self.height * 0.5 };
    }

    pub fn contains(self: Rect, point: Vec2) bool {
        return point.x >= self.x and
            point.x <= self.x + self.width and
            point.y >= self.y and
            point.y <= self.y + self.height;
    }

    pub fn containsRect(self: Rect, other: Rect) bool {
        return self.contains(.{ .x = other.x, .y = other.y }) and
            self.contains(.{ .x = other.x + other.width, .y = other.y }) and
            self.contains(.{ .x = other.x, .y = other.y + other.height }) and
            self.contains(.{ .x = other.x + other.width, .y = other.y + other.height });
    }

    pub fn intersect(self: Rect, other: Rect) ?Rect {
        const x1 = @max(self.x, other.x);
        const y1 = @max(self.y, other.y);
        const x2 = @min(self.x + self.width, other.x + other.width);
        const y2 = @min(self.y + self.height, other.y + other.height);

        if (x2 > x1 and y2 > y1) {
            return .{ .x = x1, .y = y1, .width = x2 - x1, .height = y2 - y1 };
        } else {
            return null; // No intersection
        }
    }

    pub fn merge(self: Rect, other: Rect) Rect {
        const x1 = @min(self.x, other.x);
        const y1 = @min(self.y, other.y);
        const x2 = @max(self.x + self.width, other.x + other.width);
        const y2 = @max(self.y + self.height, other.y + other.height);

        return .{ .x = x1, .y = y1, .width = x2 - x1, .height = y2 - y1 };
    }
};

pub const Vec2 = struct {
    x: f32,
    y: f32,

    pub fn zero() Vec2 {
        return .{ .x = 0, .y = 0 };
    }

    pub fn add(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = self.x + other.x, .y = self.y + other.y };
    }

    pub fn addAssign(self: *Vec2, other: Vec2) *Vec2 {
        self.x += other.x;
        self.y += other.y;
        return self;
    }

    pub fn sub(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = self.x - other.x, .y = self.y - other.y };
    }

    pub fn subAssign(self: *Vec2, other: Vec2) *Vec2 {
        self.x -= other.x;
        self.y -= other.y;
        return self;
    }

    pub fn mul(self: Vec2, scalar: f32) Vec2 {
        return .{ .x = self.x * scalar, .y = self.y * scalar };
    }

    pub fn mulAssign(self: *Vec2, scalar: f32) *Vec2 {
        self.x *= scalar;
        self.y *= scalar;
        return self;
    }

    pub fn div(self: Vec2, scalar: f32) Vec2 {
        return .{ .x = self.x / scalar, .y = self.y / scalar };
    }

    pub fn divAssign(self: *Vec2, scalar: f32) *Vec2 {
        self.x /= scalar;
        self.y /= scalar;
        return self;
    }

    pub fn length(self: Vec2) f32 {
        return @sqrt(self.x * self.x + self.y * self.y);
    }

    pub fn normalize(self: Vec2) Vec2 {
        const len = self.length();
        if (len == 0) return self; // Avoid division by zero
        return self.div(len);
    }

    pub fn normalizeAssign(self: *Vec2) *Vec2 {
        const len = self.length();
        if (len == 0) return self; // Avoid division by zero
        return self.divAssign(len);
    }

    pub fn dot(self: Vec2, other: Vec2) f32 {
        return self.x * other.x + self.y * other.y;
    }

    pub fn cross(self: Vec2, other: Vec2) f32 {
        return self.x * other.y - self.y * other.x;
    }

    pub fn distance2(self: Vec2, other: Vec2) f32 {
        const x = (other.x - self.x);
        const y = (other.y - self.y);
        return x * x + y * y;
    }

    pub fn distance(self: Vec2, other: Vec2) f32 {
        return @sqrt(self.distance2(other));
    }
};

pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn zero() Vec3 {
        return .{ .x = 0, .y = 0, .z = 0 };
    }

    pub fn add(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x + other.x, .y = self.y + other.y, .z = self.z + other.z };
    }

    pub fn addAssign(self: *Vec3, other: Vec3) *Vec3 {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        return self;
    }

    pub fn sub(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x - other.x, .y = self.y - other.y, .z = self.z - other.z };
    }

    pub fn subAssign(self: *Vec3, other: Vec3) *Vec3 {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        return self;
    }

    pub fn mul(self: Vec3, scalar: f32) Vec3 {
        return .{ .x = self.x * scalar, .y = self.y * scalar, .z = self.z * scalar };
    }

    pub fn mulAssign(self: *Vec3, scalar: f32) *Vec3 {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        return self;
    }

    pub fn div(self: Vec3, scalar: f32) Vec3 {
        return .{ .x = self.x / scalar, .y = self.y / scalar, .z = self.z / scalar };
    }

    pub fn divAssign(self: *Vec3, scalar: f32) *Vec3 {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        return self;
    }

    pub fn length(self: Vec3) f32 {
        return @sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }

    pub fn normalize(self: Vec3) Vec3 {
        const len = self.length();
        if (len == 0) return self; // Avoid division by zero
        return self.div(len);
    }

    pub fn normalizeAssign(self: *Vec3) *Vec3 {
        const len = self.length();
        if (len == 0) return self; // Avoid division by zero
        return self.divAssign(len);
    }

    pub fn dot(self: Vec3, other: Vec3) f32 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    pub fn cross(self: Vec3, other: Vec3) Vec3 {
        return .{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    pub fn distance2(self: Vec3, other: Vec3) f32 {
        const x = (other.x - self.x);
        const y = (other.y - self.y);
        const z = (other.z - self.z);
        return x * x + y * y + z * z;
    }

    pub fn distance(self: Vec3, other: Vec3) f32 {
        return @sqrt(self.distance2(other));
    }
};
